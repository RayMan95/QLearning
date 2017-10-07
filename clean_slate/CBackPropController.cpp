#include "CBackPropController.h"

CBackPropController::CBackPropController(HWND hwndMain) : CContController(hwndMain)
{
	// TOOD
	_neuralnet = new CNeuralNet(1,1,1,0.1,0.1); // ?
}

void CBackPropController::InitializeLearningAlgorithm(void)
{
	// TODO
}

bool CBackPropController::Update(void)
{
	// TODO
	return false;
}

CBackPropController::~CBackPropController(void)
{
	// TODO
}
