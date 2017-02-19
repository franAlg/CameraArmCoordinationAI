using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LaserBrazo : MonoBehaviour {

	private RaycastHit hit;

	void FixedUpdate () {
		Physics.Raycast (transform.position, transform.up, out hit);
		Debug.DrawLine (transform.position, hit.point);
		print ("El brazo apunta al punto (" + hit.point.x + ", " + hit.point.y + ", " + hit.point.z + ")");
	}

	public Vector3 getImpactPoint()
	{
		return hit.point;
	}
}
