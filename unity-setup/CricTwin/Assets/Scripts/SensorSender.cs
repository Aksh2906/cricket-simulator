using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System;
using TMPro;

public class SensorSender : MonoBehaviour
{
    public string serverIP = "192.168.0.102";  // CHANGE to PC IP
    public int port = 5055;

    public string newIP; // For UI input
    public TMP_Text ipInputField; // Reference to UI input field

    private UdpClient client;
    private long seq = 0;

    void Start()
    {
        client = new UdpClient();
        Input.gyro.enabled = true;
        Application.targetFrameRate = 60;

    }

    void Update()
    {
        Quaternion q = Input.gyro.attitude;
        Vector3 angularVel = Input.gyro.rotationRateUnbiased;

        long timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        seq++;

        string msg =
            seq + "|" +
            timestamp + "|" +
            q.x + "," + q.y + "," + q.z + "," + q.w + "|" +
            angularVel.x + "," + angularVel.y + "," + angularVel.z;

        byte[] data = Encoding.UTF8.GetBytes(msg);
        client.Send(data, data.Length, serverIP, port);
    }

    void OnApplicationQuit()
    {
        client.Close();
    }
    public void changeIP()
    {
        newIP = ipInputField.text.Trim();
        serverIP = newIP;
    }
}
