using UnityEngine;

public class BallPhysicsController : MonoBehaviour
{
    private Rigidbody rb;

    public float bounceFactor = 0.6f;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    public void Initialize(float speed, string line, string length)
    {
        Vector3 direction = GetDirection(line);
        Vector3 launchVector = direction * speed;
        launchVector.y += GetLengthHeight(length);


        // Add slight downward arc
        launchVector.y = GetLengthHeight(length);

        rb.linearVelocity = launchVector;
    }

    Vector3 GetDirection(string line)
    {
        switch (line.ToLower())
        {
            case "off":
                return new Vector3(-0.2f, 0f, -1f);
            case "middle":
                return new Vector3(0f, 0f, -1f);
            case "leg":
                return new Vector3(0.2f, 0f, -1f);
            default:
                return Vector3.forward * -1f;
        }
    }

    float GetLengthHeight(string length)
    {
        switch (length.ToLower())
        {
            case "yorker": return -8f;
            case "full": return -6f;
            case "good": return -4f;
            case "short": return 6f;
            default: return -3f;
        }
    }


}
