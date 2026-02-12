using UnityEngine;
using System.IO;

[System.Serializable]
public class BallData
{
    public float speedForce;
    public float lineX;
    public float lengthZ;
    public float swingForce;
}

public class BallReceiver : MonoBehaviour
{
    public string fileName = "final_ball.json";
    public Transform spawnPoint;
    public GameObject ballPrefab;

    void Update()
    {
        string path = Path.Combine(Application.persistentDataPath, fileName);

        if (File.Exists(path))
        {
            try
            {
                string json = File.ReadAllText(path);

                if (!string.IsNullOrEmpty(json))
                {
                    BallData data = JsonUtility.FromJson<BallData>(json);

                    SpawnBall(data);

                    File.Delete(path);
                }
            }
            catch (IOException)
            {
                // File might still be writing — skip this frame
            }
            catch (System.Exception e)
            {
                Debug.LogError("JSON Error: " + e.Message);
                File.Delete(path); // delete corrupted file
            }
        }
    }


    void SpawnBall(BallData data)
    {
        GameObject ball = Instantiate(ballPrefab, spawnPoint.position, Quaternion.identity);
        Rigidbody rb = ball.GetComponent<Rigidbody>();

        Vector3 target = new Vector3(data.lineX, 0, data.lengthZ);
        Vector3 direction = (target - spawnPoint.position).normalized;

        rb.AddForce(direction * data.speedForce, ForceMode.Impulse);

        // Apply swing force gradually
        ball.AddComponent<SwingEffect>().Init(data.swingForce);
    }
}
