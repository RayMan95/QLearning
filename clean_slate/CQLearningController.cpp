/**
(
(     )\ )
( )\   (()/(   (    ) (        (        (  (
)((_)   /(_)) ))\( /( )(   (   )\  (    )\))(
((_)_   (_))  /((_)(_)|()\  )\ |(_) )\ )((_))\
/ _ \  | |  (_))((_)_ ((_)_(_/((_)_(_/( (()(_)
| (_) | | |__/ -_) _` | '_| ' \)) | ' \)) _` |
\__\_\ |____\___\__,_|_| |_||_||_|_||_|\__, |
|___/

Refer to Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8. 3-4 (1992): 279-292
for a detailed discussion on Q Learning
*/
#include "CQLearningController.h"

CQLearningController::CQLearningController(HWND hwndMain) :
	CDiscController(hwndMain),
	_grid_size_x(CParams::WindowWidth / CParams::iGridCellDim + 1),
	_grid_size_y(CParams::WindowHeight / CParams::iGridCellDim + 1)
{
	_q_table = new Q_Table(_grid_size_x, _grid_size_y);
}

CQLearningController::~CQLearningController(void)
{
	delete _q_table;
}

/**
Chooses the next direction based on highest historic reward
*/
ROTATION_DIRECTION CQLearningController::ChooseSweeperDirection(uint SweeperNo)
{
	SVector2D<int> CurrentPos = m_vecSweepers[SweeperNo]->Position();
	int grid_x = CurrentPos.x, grid_y = CurrentPos.y;

	double q_vals[4] = { GetQ(ROTATION_DIRECTION::NORTH, grid_x, grid_y), 
		GetQ(ROTATION_DIRECTION::SOUTH, grid_x, grid_y), GetQ(ROTATION_DIRECTION::WEST, grid_x, grid_y), 
		GetQ(ROTATION_DIRECTION::EAST, grid_x, grid_y) };

	double north_q_val = q_vals[0], south_q_val = q_vals[1], west_q_val = q_vals[2], east_q_val = q_vals[3];

	if (north_q_val == 0 && south_q_val == 0 && west_q_val == 0 && east_q_val == 0) {
		// none set yet, so return random
		return ROTATION_DIRECTION(RandInt(0, 3));
	}

	ROTATION_DIRECTION max_dirs[4] = {};

	double max_q = north_q_val;
	max_dirs[0] = ROTATION_DIRECTION::NORTH;
	int index = 1;

	double q_val = south_q_val;
	if (q_val >= max_q)
	{
		if (q_val > max_q) {
			max_q = q_val;
			max_dirs[0] = ROTATION_DIRECTION::SOUTH;
			index = 1;
		}
		else // q_val == max_q; add to dirs
		{
			max_dirs[index] = ROTATION_DIRECTION::SOUTH;
			++index;
		}
	}

	q_val = west_q_val;
	if (q_val >= max_q)
	{
		if (q_val > max_q) {
			max_q = q_val;
			max_dirs[0] = ROTATION_DIRECTION::WEST;
			index = 1;
		}
		else
		{
			max_dirs[index] = ROTATION_DIRECTION::WEST;
			++index;
		}
	}

	q_val = east_q_val;
	if (q_val >= max_q)
	{
		if (q_val > max_q) {
			max_q = q_val;
			max_dirs[0] = ROTATION_DIRECTION::EAST;
			index = 1;
		}
		else
		{
			max_dirs[index] = ROTATION_DIRECTION::EAST;
			++index;
		}
	}

	--index;
	return max_dirs[RandInt(0,index)];
}

double& CQLearningController::CalculateQ(uint curr_x, uint curr_y, uint sweeper_no, double& old_Q)
{

	// Formula: Q(oldstate, action) = ((1-learnRate)*Q(oldstate, action) + (learnRate*(R + (gamma * Q(newstate, bestaction)))
	double * Q = new double(0);

	// With learning rate
	*Q = (1 - _learnRate) * old_Q;
	double new_state_best_q = GetQ(ChooseSweeperDirection(sweeper_no), curr_x, curr_y); // get best Q of current state
	*Q += _learnRate * (R(curr_x, curr_y, sweeper_no) + (_gamma * new_state_best_q));

	return *Q;
}

/**
The update method should allocate a Q table for each sweeper (this can
be allocated in one shot - use an offset to store the tables one after the other)

You can also use a boost multiarray if you wish
*/
void CQLearningController::InitializeLearningAlgorithm(void)
{
	// _num_sweepers of Q_Tables
	//_q_tables = vector<Q_Table>(_num_sweepers, Q_Table(_grid_size_x, _grid_size_y));
}
/**
The immediate reward function. This computes a reward upon achieving the goal state of
collecting all the mines on the field. It may also penalize movement to encourage exploring all directions and
of course for hitting supermines/rocks!
*/
double& CQLearningController::R(uint x, uint y, uint sweeper_no) {
	//see if it's found a mine
	// TODO final state

	int GrabHit = ((m_vecSweepers[sweeper_no])->CheckForObject(m_vecObjects,
		CParams::dMineScale));

	double * r = new double(0);

	if (GrabHit >= 0)
	{
		switch (m_vecObjects[GrabHit]->getType()) {
			case CDiscCollisionObject::Mine:
			{
				if (m_vecObjects[GrabHit]->isDead()) *r = -2; // mine already dead
				else *r = 50; // high reward for gathering mine 
				break;
			}
			case CDiscCollisionObject::Rock:
			{
				if (m_vecObjects[GrabHit]->isDead()) *r = -2; // rock already dead
				else *r = -50; // high punishment for hitting rocks
				break;
			}
			case CDiscCollisionObject::SuperMine:
			{
				if (m_vecObjects[GrabHit]->isDead()) *r = -2; // supermine already dead
				// punish for hitting supermines, but less than rocks since it clears supermine as well
				else *r = -15;
				break;
			}
		}
	}

	else *r = -2; // low punishment for moving, to promote exploring

	return *r;
}
/**
The update method. Main loop body of our Q Learning implementation
See: Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8. 3-4 (1992): 279-292
*/
bool CQLearningController::Update(void)
{
	//m_vecSweepers is the array of minesweepers
	//everything you need will be m_[something] ;)
	uint cDead = std::count_if(m_vecSweepers.begin(),
		m_vecSweepers.end(),
		[](CDiscMinesweeper * s)->bool {
		return s->isDead();
	});
	if (cDead == _num_sweepers) {
		printf("All dead ... skipping to next iteration\n");
		m_iTicks = CParams::iNumTicks;
	}

	double oldQ = 0, newQ = 0;
	SVector2D<int> oldPos, newPos;
	ROTATION_DIRECTION dir;

	vector<bool> wasAlive = vector<bool>(_num_sweepers, false);

	for (uint sw = 0; sw < _num_sweepers; ++sw) {
		if (m_vecSweepers[sw]->isDead()) continue;

		//1:::Observe the current state:
		//TODO
		wasAlive[sw] = true;
		oldPos = m_vecSweepers[sw]->Position();

		//2:::Select action with highest historic return:
		dir = ChooseSweeperDirection(sw);
		m_vecSweepers[sw]->setRotation(dir);
	}
	//now call the parents update, so all the sweepers fulfill their chosen action
	CDiscController::Update(); //call the parent's class update. Do not delete this.

	for (uint sw = 0; sw < _num_sweepers; ++sw) {
		if (m_vecSweepers[sw]->isDead() && !wasAlive[sw]) continue; // was already dead, so don't update

		//3:::Observe new state:
		oldQ = GetQ(dir, oldPos.x, oldPos.y); // get old Q
		newQ = CalculateQ(oldPos.x, oldPos.y, sw, oldQ); // calculate new Q
		
		//4:::Update _Q_s_a accordingly:
		SetQ(dir, oldPos.x, oldPos.y, newQ);
	}
	return true;
}

const double & CQLearningController::GetQ(ROTATION_DIRECTION direction, uint grid_pos_x,
	uint grid_pos_y)
{
	uint index_x = grid_pos_x / 10, index_y = grid_pos_y / 10;
	
	return _q_table->GetQ(DirectionToIndex(direction), index_x, index_y);
}

void CQLearningController::SetQ(ROTATION_DIRECTION direction, uint grid_pos_x,
	uint grid_pos_y, double new_Q)
{
	uint index_x = grid_pos_x / 10, index_y = grid_pos_y / 10;

	_q_table->SetQ(DirectionToIndex(direction), index_x, index_y, new_Q);
}

uint& CQLearningController::DirectionToIndex(ROTATION_DIRECTION direction)
{
	uint * index = new uint(0);

	switch (direction)
	{
	case ROTATION_DIRECTION::WEST:
		*index = 2;
		break;
	case ROTATION_DIRECTION::NORTH:
		*index = 1;
		break;
	case ROTATION_DIRECTION::SOUTH:
		*index = 3;
		break;
	}
	return *index;
}
