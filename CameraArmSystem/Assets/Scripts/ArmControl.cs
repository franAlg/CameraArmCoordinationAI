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

	void Start () {
		
	}

	void Update () {

		print ("rotando cabeza");
		headRotation = Quaternion.Euler(headRotationX, headRotationY, 0);
		head.rotation = Quaternion.Lerp(head.rotation, headRotation, Time.deltaTime);

		print ("rotando upper arm");
		upperRotation = Quaternion.Euler(upperRotationX, upperRotationY, 0);
		upperArm.rotation = Quaternion.Lerp(upperArm.rotation, upperRotation, Time.deltaTime);

		print ("rotando lower arm");
		lowerRotation = Quaternion.Euler(0, lowerRotationY, 0);
		lowerArm.rotation = Quaternion.Lerp(lowerArm.rotation, lowerRotation, Time.deltaTime);

	}

	public void rotateHead(float x, float y)
	{
		print ("rotando cabeza");
		headRotation = Quaternion.Euler(headRotationX, headRotationY, 0);
		head.rotation = Quaternion.Lerp(head.rotation, headRotation, Time.deltaTime);
	}

	//alfa, beta, gamma
	public void rotateArm(float upperRotationX, float upperRotationY, float lowerRotationY)
	{
		print ("rotando upper arm");
		upperRotation = Quaternion.Euler(upperRotationX, upperRotationY, 0);
		upperArm.rotation = Quaternion.Lerp(upperArm.rotation, upperRotation, Time.deltaTime);

		print ("rotando lower arm");
		lowerRotation = Quaternion.Euler(0, lowerRotationY, 0);
		lowerArm.rotation = Quaternion.Lerp(lowerArm.rotation, lowerRotation, Time.deltaTime);
	}
}
