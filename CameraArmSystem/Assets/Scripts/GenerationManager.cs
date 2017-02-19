using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GenerationManager : MonoBehaviour {
	
	private float rotateX;
	private float rotateY;

	private Vector3 angulos;

	private Vector3 resul;

	// Use this for initialization
	void Start () 
	{
		StartCoroutine (GameLoop ());
	}

	private void ResetHumanoid()
	{
		this.GetComponent<ArmControl> ().rotateHead (0.0f, 0.0f);
		this.GetComponent<ArmControl> ().rotateArm (0.0f, 0.0f, 0.0f);
	}

	private IEnumerator GameLoop()
	{
		//empezamos la nueva generacion
		yield return StartCoroutine (GenerationStarting ());
		//una vez que StartGeneration ha terminado empezamos la ejecuacion de la generacion
		yield return StartCoroutine (GenerationPlaying ());
		//una vez ha terminado la ejecucion de GenerationPlaying terminamos la generacion
		yield return StartCoroutine (GenerationEnding ());
		//volvemos a empezar
		StartCoroutine(GameLoop());
	}

	//preconfiguracion
	private IEnumerator GenerationStarting()
	{
		rotateX = Random.Range (-90.0f, 90.0f);
		rotateY = Random.Range (-180.0f, 180.0f);

		this.GetComponent<ArmControl> ().rotateHead (rotateX, rotateY);

		yield return null;
	}

	private IEnumerator GenerationPlaying()
	{
		//mandar solo los puntos de la camara, nos manda los angulos, le mandamos de nuevo ahora los dos puntos para que genere el reward?
		angulos = this.GetComponent<UDP> ().Evaluar (this.GetComponent<LaserCamara> ().getImpactPoint (), this.GetComponent<LaserBrazo> ().getImpactPoint ());

		this.GetComponent<ArmControl> ().rotateArm (angulos.x, angulos.y, angulos.z);

		resul = this.GetComponent<LaserBrazo> ().getImpactPoint ();

		this.GetComponent<UDP> ().sendResul (Vector3.Distance(this.GetComponent<LaserCamara> ().getImpactPoint (), resul));

		yield return null;
	}

	private IEnumerator GenerationEnding()
	{
		ResetHumanoid ();
		yield return null;
	}

}
