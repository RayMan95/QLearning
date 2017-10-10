#pragma once
#include "cdisccontroller.h"
#include "CParams.h"
#include "CDiscCollisionObject.h"
#include <cmath>

typedef unsigned int uint;
class CQLearningController :
	public CDiscController
{
private:
	uint _grid_size_x;
	uint _grid_size_y;
	const int _num_sweepers = CController::m_NumSweepers;
	vector<double> _q_tables = vector<double>(_num_sweepers * _grid_size_y * _grid_size_x, 0);
	ROTATION_DIRECTION ChooseSweeperDirection(SVector2D<int> CurrentPos, uint SweeperNo);
	int CalculateSweeperIndex(int x, int y, uint SweeperNo);
public:
	CQLearningController(HWND hwndMain);
	virtual void InitializeLearningAlgorithm(void);
	double R(uint x, uint y, uint sweeper_no);
	virtual bool Update(void);
	void dumpTable();
	virtual ~CQLearningController(void);
};

