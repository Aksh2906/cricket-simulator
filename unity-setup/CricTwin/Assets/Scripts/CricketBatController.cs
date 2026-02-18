using UnityEngine;
using TMPro;
using UnityEngine.InputSystem;

public class CricketBatController : MonoBehaviour
{
    [Header("UI")]
    public TMP_Text debugText;

    [Header("Rotation Settings")]
    public float rotationSmooth = 12f;

    private Quaternion baseRotation = Quaternion.identity;
    private Quaternion rotationFix = Quaternion.Euler(90f, 0f, 0f);

    private Quaternion smoothedRotation;
    private Quaternion currentRawRotation;
    void Awake()
    {
        // Optional: Auto-calibrate on start
        Calibrate();
    }

    void Start()
    {
        Calibrate();
        smoothedRotation = transform.rotation;
    }

    public void UpdateFromSensor(Quaternion gyroRot, Vector3 angularVel)
    {
        currentRawRotation = gyroRot;

        // Apply coordinate correction + calibration
        Quaternion targetRotation =
            rotationFix *
            (baseRotation * gyroRot);

        // Smooth motion
        smoothedRotation = Quaternion.Slerp(
            smoothedRotation,
            targetRotation,
            Time.deltaTime * rotationSmooth);

        transform.rotation = smoothedRotation;

        if (debugText != null)
        {
            debugText.text =
                "AngularVel: " + angularVel.ToString("F2") + "\n" +
                "Rotation: " + transform.rotation.eulerAngles.ToString("F1");
        }

        // Optional keyboard calibration
        if (Keyboard.current != null &&
            Keyboard.current.spaceKey.wasPressedThisFrame)
        {
            Calibrate();
        }
    }

    public void Calibrate()
    {
        baseRotation = Quaternion.Inverse(currentRawRotation);
        Debug.Log("Calibration complete.");
    }
}
