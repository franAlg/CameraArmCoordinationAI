using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LaserBrazo : MonoBehaviour {

	private RaycastHit hit;

	private LineRenderer line;

	void FixedUpdate () {
		line = this.GetComponent <LineRenderer> ();

		Physics.Raycast (transform.position, transform.up, out hit);
		Debug.DrawLine (transform.position, hit.point);

		line.numPositions = 2;
		line.SetPosition(0, transform.position);
		line.SetPosition(1, hit.point);
		line.startWidth = 0.02f;
		line.endWidth = 0.02f;
		line.useWorldSpace = true;

		//print ("El brazo apunta al punto (" + hit.point.x + ", " + hit.point.y + ", " + hit.point.z + ")");
	}

	public Vector3 getImpactPoint()
	{
		return hit.point;
	}
}
