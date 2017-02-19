using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GenerationManager : MonoBehaviour {

	public GameObject camara;
	public GameObject brazo;

	private float rotateX;
	private float rotateY;

	private Vector3 angulos;

	private Vector3 resul;

  public float m_StartDelay = 5f;             // The delay between the start of RoundStarting and RoundPlaying phases.
	public float m_PlayDelay = 5f;               // The delay between the end of RoundPlaying and RoundEnding phases.
  public float m_EndDelay = 5f;               // The delay between the end of RoundPlaying and RoundEnding phases.

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

		yield return m_StartDelay;
	}

	private IEnumerator GenerationPlaying()
	{
		//mandar solo los puntos de la camara, nos manda los angulos, le mandamos de nuevo ahora los dos puntos para que genere el reward?
		angulos = this.GetComponent<UDP> ().Evaluar (camara.GetComponent<LaserCamara> ().getImpactPoint (), brazo.GetComponent<LaserBrazo> ().getImpactPoint ());

		this.GetComponent<ArmControl> ().rotateArm (angulos.x, angulos.y, angulos.z);

		resul = brazo.GetComponent<LaserBrazo> ().getImpactPoint ();

		this.GetComponent<UDP> ().sendResul (Vector3.Distance(camara.GetComponent<LaserCamara> ().getImpactPoint (), resul));

		yield return m_PlayDelay;
	}

	private IEnumerator GenerationEnding()
	{
		ResetHumanoid ();
		yield return m_EndDelay;
	}

}
