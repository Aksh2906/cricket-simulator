using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Globalization;

public class HandHeightReceiver : MonoBehaviour
{
    public int port = 5060;
    public float heightScale = 2f;
    public float heightSmooth = 8f;

    private UdpClient listener;
    private Thread receiveThread;
    private bool running = true;

    private float receivedHeight = 0.5f;
    private float smoothedHeight = 0.5f;

    void Start()
    {
        listener = new UdpClient(port);
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    void Update()
    {
        smoothedHeight = Mathf.Lerp(
            smoothedHeight,
            receivedHeight,
            Time.deltaTime * heightSmooth
        );

        Vector3 pos = transform.position;
        pos.y = smoothedHeight * heightScale;
        transform.position = pos;
    }

    void ReceiveLoop()
    {
        IPEndPoint ep = new IPEndPoint(IPAddress.Any, 0);

        while (running)
        {
            byte[] data = listener.Receive(ref ep);
            string msg = Encoding.UTF8.GetString(data);

            if (float.TryParse(msg, NumberStyles.Float, CultureInfo.InvariantCulture, out float h))
            {
                receivedHeight = Mathf.Clamp01(h);
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
