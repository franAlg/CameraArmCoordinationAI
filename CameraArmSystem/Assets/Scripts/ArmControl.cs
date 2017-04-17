using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ArmControl : MonoBehaviour {

	public Transform upperArm;
	public Transform lowerArm;
	public Transform head;

	[Header("Speed")]

	public float Speed = 500.0f;

	[Header("Cabeza")]
	//angulo Fi
	public int headRotationX;
	//angulo Teta
	public int headRotationY;

	[Header("Brazo")]
	//angulo Alfa
	public int upperRotationX;
	//angulo Beta
	public int upperRotationY;
	//angulo Gamma
	public int lowerRotationY;

	[Header("Angles_UI")]
	public Text text;

//---

	private Quaternion upperRotation;
	private Quaternion lowerRotation;
	private Quaternion headRotation;

	private bool auxHead = false;
	private bool auxArm = false;

	private bool firstR = true;
	private bool firstReset = true;

	void Update () {
		//print ("rotando cabeza");
		headRotation = Quaternion.Euler(headRotationX, headRotationY, 0);
		head.rotation = Quaternion.Lerp(head.rotation, headRotation, Time.deltaTime*Speed);

		//print ("rotando upper arm");
		upperRotation = Quaternion.Euler(upperRotationX, upperRotationY, 0);
		upperArm.rotation = Quaternion.Lerp(upperArm.rotation, upperRotation, Time.deltaTime*Speed);

		//print ("rotando lower arm");
		lowerRotation = Quaternion.Euler(0, lowerRotationY, 0);
		lowerArm.rotation = Quaternion.Lerp(lowerArm.rotation, lowerRotation, Time.deltaTime*Speed);
	}

	public bool resetArm()
	{
		upperRotationX = 0;
		upperRotationY = 0;
		lowerRotationY = 0;

		if((Quaternion.Angle(Quaternion.Euler(upperRotationX, upperRotationY, 0.0f), Quaternion.Euler(upperArm.transform.eulerAngles)) < 0.1f) &&
			 (Quaternion.Angle(Quaternion.Euler(0.0f, lowerRotationY, 0.0f), Quaternion.Euler(lowerArm.transform.eulerAngles)) < 0.1f))
			{
				return true;
			}

		return false;
	}

	public bool rotateHead(int x, int y)
	{

		//print ("rotando cabeza");
		headRotationX = x;
		headRotationY = y;

		// while(!aux)
		// {
			//print("distancia head : " +  Quaternion.Angle(Quaternion.Euler(headRotationX, headRotationY, 0.0f), Quaternion.Euler(head.transform.eulerAngles)));
			if (Quaternion.Angle(Quaternion.Euler(headRotationX, headRotationY, 0.0f), Quaternion.Euler(head.transform.eulerAngles)) < 0.1f)
				return true;
		// }

		return false;
	}

	//alfa, beta, gamma
	public bool rotateArm(int upperRotX, int upperRotY, int lowerRotY)
	{
		//positivo hacia abajo, negativo hacia arriba(inverso)
		//positivo derecha, negativo izda

		//print ("rotando upper arm");
		if(firstR)
		{
			if (upperRotationX >= -90 && upperRotationX <= 90 && upperRotationY >= 0 && upperRotationY <= 90 && lowerRotationY >= -90 && lowerRotationY <= 90)
			{
				if ((upperRotationX == -90 && upperRotX == -1) || (upperRotationX == 90 && upperRotX == 1) ||
				 		(upperRotationY == 0 && upperRotY == -1) || (upperRotationY == 90 && upperRotY == 1) ||
				 		(lowerRotationY == -90 && lowerRotY == -1) || (lowerRotationY == 90 && lowerRotY == 1))
				{
					//do nothing
				}
				else {
					//print ("uX = " + upperRotX);
					upperRotationX += upperRotX;
					upperRotationY += upperRotY;

					//print ("rotando lower arm");
					lowerRotationY += lowerRotY;
				}
			}

			text.text = "Alfa:       " + upperRotationX + "\nBeta:      " + upperRotationY + "\nGamma: " + lowerRotationY;
			firstR = false;
		}



		// while(!aux)
		// {
			if((Quaternion.Angle(Quaternion.Euler(upperRotationX, upperRotationY, 0.0f), Quaternion.Euler(upperArm.transform.eulerAngles)) < 0.1f) &&
				 (Quaternion.Angle(Quaternion.Euler(0.0f, lowerRotationY, 0.0f), Quaternion.Euler(lowerArm.transform.eulerAngles)) < 0.1f))
				{
					firstR = true;
					return true;
				}
		// }

		return false;
	}
}
