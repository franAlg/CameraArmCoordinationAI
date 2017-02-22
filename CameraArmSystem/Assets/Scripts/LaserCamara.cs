using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LaserCamara : MonoBehaviour {

	private RaycastHit hit;

	void FixedUpdate () {
		Physics.Raycast (transform.position, transform.forward, out hit);
		Debug.DrawLine (transform.position, hit.point);
		//print ("La camara apunta al punto (" + hit.point.x + ", " + hit.point.y + ", " + hit.point.z + ")");
	}

	public Vector3 getImpactPoint()
	{
		return hit.point;
	}
}
