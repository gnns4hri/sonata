//******************************************************************
// 
//  Generated by RoboCompDSL
//  
//  File name: JoystickAdapter.ice
//  Source: JoystickAdapter.idsl
//
//******************************************************************
#ifndef ROBOCOMPJOYSTICKADAPTER_ICE
#define ROBOCOMPJOYSTICKADAPTER_ICE
module RoboCompJoystickAdapter
{
	struct AxisParams
	{
		bool clicked;
		string name;
		float value;
	};
	struct ButtonParams
	{
		string name;
		int step;
	};
	sequence <AxisParams> AxisList;
	sequence <ButtonParams> ButtonsList;
	struct TData
	{
		string id;
		AxisList axes;
		ButtonsList buttons;
		int velAxisIndex;
		int dirAxisIndex;
	};
	interface JoystickAdapter
	{
		idempotent void sendData (TData data);
	};
};

#endif
