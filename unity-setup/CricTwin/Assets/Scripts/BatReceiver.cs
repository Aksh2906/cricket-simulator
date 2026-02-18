using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Globalization;

public class BatReceiver : MonoBehaviour
{
    public int port = 5055;

    private UdpClient listener;
    private Thread receiveThread;
    private bool running = true;

    private readonly object dataLock = new object();
    private Quaternion latestRotation;
    private Vector3 latestAngularVel;
    private bool dataReady = false;

    void Start()
    {
        listener = new UdpClient(port);
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();

        Debug.Log("Listening on port 5055...");

    }

    void Update()
    {
        if (!dataReady) return;

        lock (dataLock)
        {
            GetComponent<CricketBatController>()
                .UpdateFromSensor(latestRotation, latestAngularVel);
        }
    }

    void ReceiveLoop()
    {
        IPEndPoint ep = new IPEndPoint(IPAddress.Any, 0);

        while (running)
        {
            byte[] data = listener.Receive(ref ep);
            string msg = Encoding.UTF8.GetString(data);
            
            string[] parts = msg.Split('|');
            if (parts.Length < 4) continue;

            string[] q = parts[2].Split(',');
            string[] w = parts[3].Split(',');

            float qx = float.Parse(q[0], CultureInfo.InvariantCulture);
            float qy = float.Parse(q[1], CultureInfo.InvariantCulture);
            float qz = float.Parse(q[2], CultureInfo.InvariantCulture);
            float qw = float.Parse(q[3], CultureInfo.InvariantCulture);

            float wx = float.Parse(w[0], CultureInfo.InvariantCulture);
            float wy = float.Parse(w[1], CultureInfo.InvariantCulture);
            float wz = float.Parse(w[2], CultureInfo.InvariantCulture);

            Quaternion unityRot = new Quaternion(qx, qy, -qz, -qw);
            Debug.Log("Packet received!");

            lock (dataLock)
            {
                latestRotation = unityRot;
                latestAngularVel = new Vector3(wx, wy, wz);
                dataReady = true;
            }
        }
    }

    void OnApplicationQuit()
    {
        running = false;
        listener.Close();
        receiveThread.Abort();
    }
}
