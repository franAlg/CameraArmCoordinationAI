using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LaserCamara : MonoBehaviour {

	private RaycastHit hit;

	private LineRenderer line;

	void FixedUpdate () {
		line = this.GetComponent <LineRenderer> ();

		Physics.Raycast (transform.position, transform.forward, out hit);
		Debug.DrawLine (transform.position, hit.point);

		line.numPositions = 2;
		line.SetPosition(0, transform.position);
		line.SetPosition(1, hit.point);
		line.startWidth = 0.02f;
		line.endWidth = 0.02f;
		line.useWorldSpace = true;

		//print ("La camara apunta al punto (" + hit.point.x + ", " + hit.point.y + ", " + hit.point.z + ")");
	}

	public Vector3 getImpactPoint()
	{
		return hit.point;
	}
}
