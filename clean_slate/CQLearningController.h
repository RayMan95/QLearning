#pragma once
#include "cdisccontroller.h"
#include "CParams.h"
#include "CDiscCollisionObject.h"
#include <cmath>

typedef unsigned int uint;

struct Q_Table {
private:
	vector<vector<vector<double>>> _q_table;
public:
	// 4 coz 4 directions
	Q_Table(uint x, uint y) { 
		_q_table = vector<vector<vector<double>>>(y, vector<vector<double>>(x, vector<double>(4, 0.0))); 
	}

	const double& GetQ(uint direction, uint index_x, uint index_y) const
	{
		return _q_table[index_y][index_x][direction];
	}

	void SetQ(uint direction, uint index_x, uint index_y, double& new_Q)
	{
		_q_table[index_y][index_x][direction] = new_Q;
	}
};

class CQLearningController :
	public CDiscController
{
private:
	uint _grid_size_x;
	uint _grid_size_y;
	const double _gamma = 0.8; // future reward discount (gamma)
	const double _learnRate = 0.2; // learning rate (alpha)
	double _mines_gathered = 0;
	const uint _num_sweepers = CController::m_NumSweepers;
	Q_Table * _q_table;
	ROTATION_DIRECTION ChooseSweeperDirection(uint SweeperNo);
	double& CalculateQ(ROTATION_DIRECTION direction, uint curr_x, uint curr_y, uint sweeper_no, double& old_Q);
	const double& GetQ(ROTATION_DIRECTION direction, uint grid_pos_x, uint grid_pos_y);
	void SetQ(ROTATION_DIRECTION direction, uint grid_pos_x, uint grid_pos_y,
		double new_Q);
	uint& DirectionToIndex(ROTATION_DIRECTION direction);
public:
	CQLearningController(HWND hwndMain);
	virtual void InitializeLearningAlgorithm(void);
	double& R(uint x, uint y, uint sweeper_no);
	virtual bool Update(void);
	virtual ~CQLearningController(void);
};

