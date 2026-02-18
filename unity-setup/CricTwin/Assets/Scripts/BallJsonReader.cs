using UnityEngine;
using System.IO;
using System.Collections;

public class BallJsonReader : MonoBehaviour
{
    public string fileName = "ball.json";
    public BallQueueManager queueManager;

    private BallData lastBall = null;

    void Start()
    {
        StartCoroutine(CheckJsonLoop());
    }

    IEnumerator CheckJsonLoop()
    {
        while (true)
        {
            string path = Path.Combine(Application.streamingAssetsPath, fileName);

            if (File.Exists(path))
            {
                string json = File.ReadAllText(path);

                try
                {
                    BallData data = JsonUtility.FromJson<BallData>(json);

                    if (IsNewBall(data))
                    {
                        queueManager.AddBall(data);
                        lastBall = data;

                        Debug.Log("New ball detected and queued.");
                    }
                }
                catch
                {
                    Debug.Log("JSON read error — waiting for next update.");
                }
            }

            yield return new WaitForSeconds(0.5f);
        }
    }

    bool IsNewBall(BallData newBall)
    {
        if (lastBall == null)
            return true;

        return newBall.speed != lastBall.speed ||
               newBall.line != lastBall.line ||
               newBall.length != lastBall.length;
    }
}
