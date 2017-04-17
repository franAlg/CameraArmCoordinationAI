using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

public class GenerationManager : MonoBehaviour {


	public GameObject camara;
	public GameObject brazo;

	[Header("Delays")]

  public float m_StartDelay = 1.0f;             // The delay between the start of RoundStarting and RoundPlaying phases.
	public float m_PlayDelay = 1.0f;               // The delay between the end of RoundPlaying and RoundEnding phases.
  public float m_EndDelay = 1.0f;               // The delay between the end of RoundPlaying and RoundEnding phases.

	[Header("Distance_UI")]

	public Text text;

//---

	private int rotateX = 1;
	private int rotateY = 1;

	private Vector3 angulos;
	private Vector3 resul;

	private bool wait = false;
	private int i = 0;

	private bool finishedH = false;
	private bool finishedA = false;

	private bool newEp = false;



	// Use this for initialization
	void Start ()
	{
		text.text = "Distancia: ";

		this.GetComponent<UDP> ().init();

		StartCoroutine (GameLoop ());
	}

	// private void init()
	// {
	// 	this.GetComponent<UDP> ().init();
	// 	StartCoroutine (GameLoop ());
	// }

	private IEnumerator GameLoop()
	{
		//print("Inicio bucle " + i);

		//empezamos la nueva generacion
		yield return StartCoroutine (GenerationStarting ());
		//una vez que StartGeneration ha terminado empezamos la ejecuacion de la generacion
		yield return StartCoroutine (GenerationPlaying ());
		//una vez ha terminado la ejecucion de GenerationPlaying terminamos la generacion
		yield return StartCoroutine (GenerationEnding ());
		//print("Fin bucle " + i);
		i++;
		//volvemos a empezar
		StartCoroutine(GameLoop());
	}

	//preconfiguracion
	private IEnumerator GenerationStarting()
	{
		//Mientras se espera a que acaben su movimiento el brazo y la cabeza se hacen iteraciones y se lee del socket, haciendo que python vaya 3 veces por delante que unity

		//solo falta arreglar aqui
		 if (this.GetComponent<UDP> ().nuevoEpisodio())
		 {
			 newEp = true;
			rotateX = Random.Range (-40, 41);
			rotateY = Random.Range (-40, 41);

			while(!this.GetComponent<ArmControl> ().rotateHead (rotateX, rotateY))
					yield return null;

			while(!this.GetComponent<ArmControl> ().resetArm ())
					yield return null;
		 }

		 finishedH = true;
		//  Debug.Log("rotateX: " + rotateX);
		//  Debug.Log("rotateY: " + rotateY);

	}

	private IEnumerator GenerationPlaying()
	{
		if(newEp)
		{
			angulos = this.GetComponent<UDP> ().EvaluaNuevoEp (camara.GetComponent<LaserCamara> ().getImpactPoint (), brazo.GetComponent<LaserBrazo> ().getImpactPoint ());
			newEp = false;
		}
		else {
			angulos = this.GetComponent<UDP> ().EvaluaStep ();
		}

		while(!this.GetComponent<ArmControl> ().rotateArm ((int)angulos.x, (int)angulos.y, (int)angulos.z))
			yield return null;

		finishedA = true;
	}

	private IEnumerator GenerationEnding()
	{
		int deltaAlfa, deltaBeta, deltaGamma;

		if (finishedA && finishedH)
		{
			finishedH = false;
			finishedA = false;

			resul = brazo.GetComponent<LaserBrazo> ().getImpactPoint ();

			text.text = "Distancia: " + Vector3.Distance(resul, camara.GetComponent<LaserCamara> ().getImpactPoint ());

			// +-1 grado de error en cada componente
			deltaAlfa = (int)(camara.GetComponent<LaserCamara> ().getImpactPoint ().x - resul.x);
			deltaBeta = (int)(camara.GetComponent<LaserCamara> ().getImpactPoint ().y - resul.y);
			deltaGamma = (int)(camara.GetComponent<LaserCamara> ().getImpactPoint ().z - resul.z);

			this.GetComponent<UDP> ().sendResul (deltaAlfa, deltaBeta, deltaGamma);
		}
		else yield return null;

		//yield return new WaitForSeconds(m_EndDelay);
	}

	// public void OnApplicationQuit()
	// 	{
	// 		 // end of application
	// 		 if (managerThread != null)
	// 		 {
	// 				managerThread.Abort();
	// 		 }
	// 	 }
}
