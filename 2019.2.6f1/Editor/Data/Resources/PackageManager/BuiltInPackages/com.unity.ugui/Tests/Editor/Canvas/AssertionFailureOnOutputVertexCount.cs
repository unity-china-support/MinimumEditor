using NUnit.Framework;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.TestTools;
using UnityEditor.SceneManagement;

public class AssertionFailureOnOutputVertexCount
{
    [Test]
    public void AssertionFailureOnOutputVertexCountTest()
    {
        var newScene = EditorSceneManager.NewScene(UnityEditor.SceneManagement.NewSceneSetup.DefaultGameObjects, UnityEditor.SceneManagement.NewSceneMode.Single);

        var canvasMaster = new GameObject("Canvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        var canvasChild = new GameObject("Canvas Child", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        canvasChild.transform.SetParent(canvasMaster.transform);

        var panel1 = new GameObject("Panel", typeof(CanvasRenderer), typeof(UnityEngine.UI.Image));
        panel1.transform.SetParent(canvasMaster.transform);

        var panel2 = new GameObject("Panel", typeof(CanvasRenderer), typeof(UnityEngine.UI.Image));
        panel2.transform.SetParent(canvasChild.transform);

        string scenePath = "Assets/AssertionFailureOnOutputVertexCountTestScene.unity";

        // Saving a scene would trigger the error case 893551
        EditorSceneManager.SaveScene(newScene, scenePath);
        Debug.Log("Success");

        LogAssert.Expect(LogType.Log, "Success");
    }
}
