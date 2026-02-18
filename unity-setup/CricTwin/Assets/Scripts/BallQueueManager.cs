using UnityEngine;
using System.Collections.Generic;

public class BallQueueManager : MonoBehaviour
{
    private Queue<BallData> ballQueue = new Queue<BallData>();

    public void AddBall(BallData data)
    {
        ballQueue.Enqueue(data);
    }

    public bool HasBall()
    {
        return ballQueue.Count > 0;
    }

    public BallData GetNextBall()
    {
        return ballQueue.Dequeue();
    }
}
