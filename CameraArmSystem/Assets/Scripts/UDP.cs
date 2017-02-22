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
	private Vector3 anguloArm = new Vector3(0.0f, 0.0f, 0.0f);

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

			bufRec = udpClient.Receive(ref RemoteIpEndPoint);
			mRecibir = Encoding.ASCII.GetString(bufRec);
			mRecibir = mRecibir.Replace('.', ',');
			Debug.Log("AnguloX: " + float.Parse(mRecibir));
			anguloArm.x = float.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);

			bufRec = udpClient.Receive(ref RemoteIpEndPoint);
			mRecibir = Encoding.ASCII.GetString(bufRec);
			mRecibir = mRecibir.Replace('.', ',');
			Debug.Log("AnguloY: " + float.Parse(mRecibir));
			anguloArm.y = float.Parse(mRecibir);
			Array.Clear(bufRec, 0, bufRec.Length);

			bufRec = udpClient.Receive(ref RemoteIpEndPoint);
			mRecibir = Encoding.ASCII.GetString(bufRec);
			mRecibir = mRecibir.Replace('.', ',');
			Debug.Log("AnguloZ: " + float.Parse(mRecibir));
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
			dist = Encoding.ASCII.GetBytes(distancia.ToString());

			// Sends a message to the host to which you have connected.
			udpClient.Send(dist, dist.Length);

		}catch (Exception e ) {
			Debug.Log(e.ToString());
		}

	}

}
