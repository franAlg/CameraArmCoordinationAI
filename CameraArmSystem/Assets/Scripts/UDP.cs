using UnityEngine;
using System.Collections;

using System.Net.Sockets;
using System.Net;
using System.IO;
using System.Text;
using System;

public class UDP : MonoBehaviour {

	private UdpClient udpClient;
	private IPEndPoint RemoteIpEndPoint;

	private byte[] camX;
	private byte[] camY;
	private byte[] camZ;

	private byte[] armX;
	private byte[] armY;
	private byte[] armZ;

	private byte[] dist;

	private byte[] deltaAlfa;
	private byte[] deltaBeta;
	private byte[] deltaGamma;

	//(anguloAlfa, anguloBeta, anguloGamma)
	private Vector3 anguloArm = new Vector3(0, 0, 0);

	public int clientPort; //11000
	public int serverPort; //9900

	private String mRecibir;
	private byte[] bufRec;

	private int newEpisode = 0;
	public bool isLive = false;

	// Use this for initialization
	// void Start () {
	//
	// 	udpClient = new UdpClient(clientPort);
	// 	//esto para la hora de recibir
	// 	RemoteIpEndPoint = new IPEndPoint(IPAddress.Loopback, serverPort);
	//
	// 	try{
	// 		Debug.Log("Cliente Robotic Arm conectado con el Servidor");
	// 		udpClient.Connect("localhost", serverPort);
	// 		Debug.Log("Conectado");
	// 		isLive = true;
	// 	}
	// 	catch (Exception e ) {
	// 		Debug.Log(e.ToString());
	// 	}
	//
	// }

	public void init () {

		udpClient = new UdpClient(clientPort);
		//esto para la hora de recibir
		RemoteIpEndPoint = new IPEndPoint(IPAddress.Loopback, serverPort);

		try{
			Debug.Log("Cliente Robotic Arm conectado con el Servidor");
			udpClient.Connect("localhost", serverPort);
			Debug.Log("Conectado");
			isLive = true;
		}
		catch (Exception e ) {
			Debug.Log(e.ToString());
		}

	}

	public Vector3 EvaluaNuevoEp (Vector3 cam, Vector3 arm) {
		try{
			camX = Encoding.ASCII.GetBytes(cam.x.ToString());
			camY = Encoding.ASCII.GetBytes(cam.y.ToString());
			camZ = Encoding.ASCII.GetBytes(cam.z.ToString());

			armX = Encoding.ASCII.GetBytes(arm.x.ToString());
			armY = Encoding.ASCII.GetBytes(arm.y.ToString());
			armZ = Encoding.ASCII.GetBytes(arm.z.ToString());

			// Sends a message to the host to which you have connected.
			udpClient.Send(camX, camX.Length);
			udpClient.Send(camY, camY.Length);
			udpClient.Send(camZ, camZ.Length);

			udpClient.Send(armX, armX.Length);
			udpClient.Send(armY, armY.Length);
			udpClient.Send(armZ, armZ.Length);

			// Blocks until a message returns on this socket from a remote host.
			//Debug.Log("empiezo a leer evaluar");
			bufRec = udpClient.Receive(ref RemoteIpEndPoint);
			mRecibir = Encoding.ASCII.GetString(bufRec);
			mRecibir = mRecibir.Replace('.', ',');
			//Debug.Log("AnguloX: " + float.Parse(mRecibir));
			anguloArm.x = int.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);
			//print("Angulo X : " + anguloArm.x);

			bufRec = udpClient.Receive(ref RemoteIpEndPoint);
			mRecibir = Encoding.ASCII.GetString(bufRec);
			mRecibir = mRecibir.Replace('.', ',');
			//Debug.Log("AnguloY: " + float.Parse(mRecibir));
			anguloArm.y = int.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);

			bufRec = udpClient.Receive(ref RemoteIpEndPoint);
			mRecibir = Encoding.ASCII.GetString(bufRec);
			mRecibir = mRecibir.Replace('.', ',');
			//Debug.Log("AnguloZ: " + float.Parse(mRecibir));
			anguloArm.z = int.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);


		}catch (Exception e ) {
			Debug.Log(e.ToString());
			anguloArm.x = 0;
			anguloArm.y = 0;
			anguloArm.z = 0;
		}

		return anguloArm;
	}

	public Vector3 EvaluaStep () {
		try{

			// Blocks until a message returns on this socket from a remote host.
			//Debug.Log("empiezo a leer evaluar");
			bufRec = udpClient.Receive(ref RemoteIpEndPoint);
			mRecibir = Encoding.ASCII.GetString(bufRec);
			mRecibir = mRecibir.Replace('.', ',');
			//Debug.Log("AnguloX: " + float.Parse(mRecibir));
			anguloArm.x = int.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);
			//print("Angulo X : " + anguloArm.x);

			bufRec = udpClient.Receive(ref RemoteIpEndPoint);
			mRecibir = Encoding.ASCII.GetString(bufRec);
			mRecibir = mRecibir.Replace('.', ',');
			//Debug.Log("AnguloY: " + float.Parse(mRecibir));
			anguloArm.y = int.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);

			bufRec = udpClient.Receive(ref RemoteIpEndPoint);
			mRecibir = Encoding.ASCII.GetString(bufRec);
			mRecibir = mRecibir.Replace('.', ',');
			//Debug.Log("AnguloZ: " + float.Parse(mRecibir));
			anguloArm.z = int.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);


		}catch (Exception e ) {
			Debug.Log(e.ToString());
			anguloArm.x = 0;
			anguloArm.y = 0;
			anguloArm.z = 0;
		}

		return anguloArm;
	}

	public void sendResul (int alfa, int beta, int gamma) {
		try{

			deltaAlfa = Encoding.ASCII.GetBytes(alfa.ToString());
			deltaBeta = Encoding.ASCII.GetBytes(beta.ToString());
			deltaGamma = Encoding.ASCII.GetBytes(gamma.ToString());

			// Sends a message to the host to which you have connected.
			udpClient.Send(deltaAlfa, deltaAlfa.Length);
			udpClient.Send(deltaBeta, deltaBeta.Length);
			udpClient.Send(deltaGamma, deltaGamma.Length);

		}catch (Exception e ) {
			Debug.Log(e.ToString());
		}

	}

	public bool nuevoEpisodio () {
		try{
				//print("empiezo a leer nuevo episodio");
				bufRec = udpClient.Receive(ref RemoteIpEndPoint);
				mRecibir = Encoding.ASCII.GetString(bufRec);
				newEpisode = int.Parse(mRecibir);
				Array.Clear(bufRec, 0, bufRec.Length);
				//print("newEpisode : " + newEpisode);

		 }catch (Exception e ) {
		 	Debug.Log(e.ToString());
		 }

		if (newEpisode == 1)
		{
			newEpisode = 0;
			return true;
		}
		else return false;
	}

	void OnApplicationQuit() {
		if (udpClient!=null)
			udpClient.Close();
	}

	public bool isUDPlive () {
			return isLive;

	}

}
