using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ArmControl : MonoBehaviour {

	public Transform upperArm;
	public Transform lowerArm;
	public Transform head;

	[Header("Cabeza")]
	//angulo Fi
	public float headRotationX;
	//angulo Teta
	public float headRotationY;

	[Header("Brazo")]
	//angulo Alfa
	public float upperRotationX;
	//angulo Beta
	public float upperRotationY;
	//angulo Gamma
	public float lowerRotationY;

	private Quaternion upperRotation;
	private Quaternion lowerRotation;
	private Quaternion headRotation;

	private bool aux = false;

	void Start () {

	}

	void Update () {

		//print ("rotando cabeza");
		headRotation = Quaternion.Euler(headRotationX, headRotationY, 0);
		head.rotation = Quaternion.Lerp(head.rotation, headRotation, Time.deltaTime);

		//print ("rotando upper arm");
		upperRotation = Quaternion.Euler(upperRotationX, upperRotationY, 0);
		upperArm.rotation = Quaternion.Lerp(upperArm.rotation, upperRotation, Time.deltaTime);

		//print ("rotando lower arm");
		lowerRotation = Quaternion.Euler(0, lowerRotationY, 0);
		lowerArm.rotation = Quaternion.Lerp(lowerArm.rotation, lowerRotation, Time.deltaTime);

	}

	public bool rotateHead(float x, float y)
	{

		//print ("rotando cabeza");
		headRotationX = x;
		headRotationY = y;

		while(!aux)
		{
			if (Vector3.Distance(new Vector3(headRotationX, headRotationY, 0.0f), head.transform.eulerAngles) > 0.1f)
				aux = true;
		}

		return true;
	}

	//alfa, beta, gamma
	public bool rotateArm(float upperRotX, float upperRotY, float lowerRotY)
	{
		//print ("rotando upper arm");
		upperRotationX = upperRotX;
		upperRotationY = upperRotY;

		//print ("rotando lower arm");
		lowerRotationY = lowerRotY;

		while(!aux)
		{
			if((Vector3.Distance(new Vector3(headRotationX, headRotationY, 0.0f), upperArm.transform.eulerAngles) > 0.1f) &&
				 (Vector3.Distance(new Vector3(0.0f, lowerRotationY, 0.0f), lowerArm.transform.eulerAngles) > 0.1f))
				 aux = true;
		}

		return true;
	}
}
