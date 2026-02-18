using UnityEngine;
using System.Collections;

public class BallGameLoop : MonoBehaviour
{
    public BallQueueManager queueManager;
    public BallSpawner spawner;

    public float spawnInterval = 3f; // match real timing

    void Start()
    {
        StartCoroutine(GameLoop());
    }

    IEnumerator GameLoop()
    {
        while (true)
        {
            if (queueManager.HasBall())
            {
                BallData data = queueManager.GetNextBall();
                spawner.SpawnBall(data);
            }

            yield return new WaitForSeconds(spawnInterval);
        }
    }
}
