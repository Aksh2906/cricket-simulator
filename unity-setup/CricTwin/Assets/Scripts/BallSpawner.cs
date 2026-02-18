using UnityEngine;

public class BallSpawner : MonoBehaviour
{
    public GameObject ballPrefab;
    public Transform spawnPoint;

    [Header("Speed Settings")]
    public float speedScale = 0.25f; // reduce real speed

    public void SpawnBall(BallData data)
    {
        GameObject ball = Instantiate(ballPrefab, spawnPoint.position, Quaternion.identity);

        BallPhysicsController controller = ball.GetComponent<BallPhysicsController>();

        float unitySpeed = data.speed * speedScale;

        controller.Initialize(unitySpeed, data.line, data.length);
    }
}
