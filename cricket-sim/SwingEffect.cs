public class SwingEffect : MonoBehaviour
{
    float swingForce;
    Rigidbody rb;

    public void Init(float force)
    {
        swingForce = force;
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        if (rb != null)
        {
            rb.AddForce(Vector3.right * swingForce, ForceMode.Force);
        }
    }
}
