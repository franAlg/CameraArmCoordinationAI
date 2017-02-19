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

	//(anguloAlfa, anguloBeta, anguloGamma)
	private Vector3 anguloArm;

	public int clientPort; //11000
	public int serverPort; //9900

	private String mRecibir;
	private byte[] bufRec;

	// Use this for initialization
	void Start () {

		udpClient = new UdpClient(clientPort);
		//esto para la hora de recibir
		RemoteIpEndPoint = new IPEndPoint(IPAddress.Loopback, serverPort);

		try{
			Debug.Log("Cliente Drone conectado con el Servidor");
			udpClient.Connect("localhost", serverPort);
			Debug.Log("Conectado");

		}  
		catch (Exception e ) {
			Debug.Log(e.ToString());
		}

	}

	//OJO SIEMPRE QUE SE LLAME A ESTA FUNCION SE TIENEN QUE PONER LOS ARGUMENTOS CON ALGUN VALOR DECIMAL EJEMPLO XXX.1
	public Vector3 Evaluar (Vector3 cam, Vector3 arm) {
		try{
			camX = BitConverter.GetBytes(cam.x);
			camY = BitConverter.GetBytes(cam.y);
			camZ = BitConverter.GetBytes(cam.z);

			armX = BitConverter.GetBytes(arm.x);
			armY = BitConverter.GetBytes(arm.y);
			armZ = BitConverter.GetBytes(arm.z);

			// Sends a message to the host to which you have connected.
			udpClient.Send(camX, camX.Length);
			udpClient.Send(camY, camY.Length);
			udpClient.Send(camZ, camZ.Length);

			udpClient.Send(armX, armX.Length);
			udpClient.Send(armY, armY.Length);
			udpClient.Send(armZ, armZ.Length);

			// Blocks until a message returns on this socket from a remote host.
			bufRec = udpClient.Receive(ref RemoteIpEndPoint); 
			mRecibir = Encoding.ASCII.GetString(bufRec);
			anguloArm.x = float.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);

			bufRec = udpClient.Receive(ref RemoteIpEndPoint); 
			mRecibir = Encoding.ASCII.GetString(bufRec);
			anguloArm.y = float.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);

			bufRec = udpClient.Receive(ref RemoteIpEndPoint); 
			mRecibir = Encoding.ASCII.GetString(bufRec);
			anguloArm.z = float.Parse(mRecibir);


		}catch (Exception e ) {
			Debug.Log(e.ToString());
			anguloArm.x = 0.0f;
			anguloArm.y = 0.0f;
			anguloArm.z = 0.0f;
		}

		return anguloArm;
	}

	public void sendResul (float distancia) {
		try{
			dist = BitConverter.GetBytes(distancia);

			// Sends a message to the host to which you have connected.
			udpClient.Send(dist, dist.Length);

		}catch (Exception e ) {
			Debug.Log(e.ToString());
		}

	}

}