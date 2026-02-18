using UnityEngine;

public class BatImpactController : MonoBehaviour
{
    public float minShotPower = 8f;
    public float maxShotPower = 25f;
    public float shotAssistMultiplier = 2.5f;

    private Vector3 lastBatVelocity;

    void Update()
    {
        // Estimate bat velocity manually
        lastBatVelocity = (transform.position - lastPosition) / Time.deltaTime;
        lastPosition = transform.position;
    }

    private Vector3 lastPosition;

    void Start()
    {
        lastPosition = transform.position;
    }

    void OnCollisionEnter(Collision collision)
    {
        if (!collision.gameObject.CompareTag("Ball")) return;

        Rigidbody ballRb = collision.rigidbody;

        Vector3 contactNormal = collision.contacts[0].normal;

        // Shot direction = forward + upward assist
        Vector3 shotDirection =
            transform.forward * 0.7f +
            transform.up * 0.3f;

        shotDirection.Normalize();

        float rawPower = lastBatVelocity.magnitude;

        // 🔥 SHOT ASSIST
        float assistedPower = Mathf.Clamp(
            rawPower * shotAssistMultiplier,
            minShotPower,
            maxShotPower
        );

        // Override ball velocity completely
        ballRb.linearVelocity = shotDirection * assistedPower;
    }
}
