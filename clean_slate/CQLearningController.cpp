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

/**
Chooses the next direction based on highest historic reward
*/
ROTATION_DIRECTION CQLearningController::ChooseSweeperDirection(SVector2D<int> CurrentPos, uint SweeperNo)
{
	int x = CurrentPos.x, y = CurrentPos.y;

	int up = y + 10, down = y - 10, left = x - 10,
		right = x + 10;
	if (right >= CParams::WindowWidth) right = 0;
	if (left < 0) left = CParams::WindowWidth - CParams::iGridCellDim;
	if (up >= CParams::WindowHeight) up = 0;
	if (down < 0) down = CParams::WindowHeight - CParams::iGridCellDim;

	// divide for index

	up /= 10;
	down /= 10;
	left /= 10;
	right /= 10;

	x /= 10;
	y /= 10;

	// so if no direction with max_q (i.e. all 0s), rotate as it would have anyway
	ROTATION_DIRECTION dir = m_vecSweepers[SweeperNo]->getRotation();
	double max_q = 0;

	if (_q_tables[CalculateSweeperIndex(x, up, SweeperNo)] > max_q)
	{
		max_q = up;
		dir = ROTATION_DIRECTION::NORTH;
	}
	if (_q_tables[CalculateSweeperIndex(x, down, SweeperNo)] > max_q)
	{
		max_q = down;
		dir = ROTATION_DIRECTION::SOUTH;
	}
	if (_q_tables[CalculateSweeperIndex(left, y, SweeperNo)] > max_q)
	{
		max_q = left;
		dir = ROTATION_DIRECTION::WEST;
	}
	if (_q_tables[CalculateSweeperIndex(right, y, SweeperNo)] > max_q)
	{
		dir = ROTATION_DIRECTION::EAST;
	}

	return dir;
}

/**
Calculates Q table index based on position
*/
int CQLearningController::CalculateSweeperIndex(int x, int y, uint SweeperNo)
{
	int i = 0, j = 0, k = 0;
	
	x /= 10;
	k = x *_grid_size_x;
	y /= 10;
	j = y * _grid_size_y;

	i = SweeperNo * _num_sweepers;

	return i + j + k;
}

CQLearningController::CQLearningController(HWND hwndMain) :
	CDiscController(hwndMain),
	_grid_size_x(CParams::WindowWidth / CParams::iGridCellDim + 1),
	_grid_size_y(CParams::WindowHeight / CParams::iGridCellDim + 1)
{
}
/**
The update method should allocate a Q table for each sweeper (this can
be allocated in one shot - use an offset to store the tables one after the other)

You can also use a boost multiarray if you wish
*/
void CQLearningController::InitializeLearningAlgorithm(void)
{
	//TODO

}
/**
The immediate reward function. This computes a reward upon achieving the goal state of
collecting all the mines on the field. It may also penalize movement to encourage exploring all directions and
of course for hitting supermines/rocks!
*/
double CQLearningController::R(uint x, uint y, uint sweeper_no) {
	//see if it's found a mine
	// TODO final state
	int GrabHit = ((m_vecSweepers[sweeper_no])->CheckForObject(m_vecObjects,
		CParams::dMineScale));

	if (GrabHit >= 0)
	{
		switch (m_vecObjects[GrabHit]->getType()) {
			case CDiscCollisionObject::Mine:
			{
				if (!m_vecObjects[GrabHit]->isDead()) return 20; // high reward for gathering mine 
				else return 2; // low reward for moving, to promote moving
				break;
			}
			case CDiscCollisionObject::Rock:
			{
				return -20; // high punishment for hitting rocks
				break;
			}
			case CDiscCollisionObject::SuperMine:
			{
				// punish for hitting supermines, but less than rocks since it clears supermine as well
				return -10;
				break;
			}
		}
	}

	else return 2; // low reward for moving, to promote moving
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

	// YOU ARE HERE
	// print out Q table
	dumpTable();

	// resets Q every time??
	/*_q_tables = new double**[_grid_size_y];

	for (int i = 0; i < sweepers; ++i) {
	_q_tables[i] = new double*[_grid_size_x];
	for (int j = 0; j < _grid_size_x; ++j) {
	_q_tables[i][j] = new double[_grid_size_y];
	for (int k = 0; k < _grid_size_y; ++k) {
	_q_tables[i][j][k] = 0;
	}
	}
	}*/



	double oldQ = 0, newQ = 0;
	SVector2D<int> oldPos, newPos;

	for (uint sw = 0; sw < _num_sweepers; ++sw) {
		if (m_vecSweepers[sw]->isDead()) continue;
		/**
		Q-learning algorithm according to:
		Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8. 3-4 (1992): 279-292
		*/
		//1:::Observe the current state:
		//TODO
		oldPos = m_vecSweepers[sw]->Position();

		SVector2D<int> index = CalculateSweeperIndex(oldPos.x, oldPos.y, sw);
		oldQ = _q_tables[CalculateSweeperIndex(oldPos.x, oldPos.y, sw)];
		//2:::Select action with highest historic return:
		//TODO
		m_vecSweepers[sw]->setRotation(ChooseSweeperDirection(oldPos, sw));
	}
	//now call the parents update, so all the sweepers fulfill their chosen action
	CDiscController::Update(); //call the parent's class update. Do not delete this.

	for (uint sw = 0; sw < _num_sweepers; ++sw) {
		if (m_vecSweepers[sw]->isDead()) continue;
		//3:::Observe new state:
		//TODO
		newPos = m_vecSweepers[sw]->Position();

		newQ = _q_tables[CalculateSweeperIndex(newPos.x, newPos.y, sw)];
		//4:::Update _Q_s_a accordingly:
		//TODO
		_q_tables[CalculateSweeperIndex(newPos.x, newPos.y, sw)] = R(newPos.x, newPos.y, sw);
	}
	return true;
}

/**
Prints out Q table
*/
void CQLearningController::dumpTable()
{
	for (int i = 0; i < _num_sweepers; ++i) {
		cout << "Sweeper #" << i << endl;
		for (int j = 0; j < _grid_size_y; ++j) {
			for (int k = 0; k < _grid_size_x; ++k) {
				cout << to_string(_q_tables[CalculateSweeperIndex(k, j, i)]) << endl;
			}
			cout << endl;
		}
		cout << "---------------------------" << endl;
	}
}

CQLearningController::~CQLearningController(void)
{
	//TODO: dealloc stuff here if you need to	
}
