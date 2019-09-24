//Entire class disabled due to a major instability - discussion on Slack at #ui-toggle-instability

/*using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.TestTools;
using UnityEngine.UI;
using UnityEngine.UI.Tests;
using Object = UnityEngine.Object;

namespace ToggleTest
{
    abstract class BaseToggleTests : IPrebuildSetup
    {
        const string kPrefabTogglePath = "Assets/Resources/TestToggle.prefab";
        const string kPrefabToggleGroupPath = "Assets/Resources/TestToggleGroup.prefab";

        protected GameObject m_PrefabRoot;
        protected List<Toggle> m_toggle = new List<Toggle>();
        protected static int nbToggleInGroup = 2;

        public void Setup()
        {
#if UNITY_EDITOR
            CreateSingleTogglePrefab();
            CreateToggleGroupPrefab();
#endif
        }

        private static void CreateSingleTogglePrefab()
        {
#if UNITY_EDITOR
            var rootGO = new GameObject("rootGo");
            GameObject canvasGO = ComponentCreator.CreateCanvasRoot("Canvas");
            canvasGO.transform.SetParent(rootGO.transform);

            // factory function alias for brevity
            var c = ComponentCreator.CreationInfo.Creator;
            ComponentCreator.CreationInfo template =
                c("TestToggle") // object name
                    .With<Toggle>()
                    .With<ToggleTestImageHook>();

            var nameToObjectMapping = new Dictionary<string, GameObject>();
            ComponentCreator.CreateHierarchy(info: template, parent: canvasGO, dict: nameToObjectMapping);

            var toggle = nameToObjectMapping["TestToggle"].GetComponent<Toggle>();
            toggle.enabled = true;
            toggle.graphic = nameToObjectMapping["TestToggle"].GetComponent<ToggleTestImageHook>();
            toggle.graphic.canvasRenderer.SetColor(Color.white);

            if (!Directory.Exists("Assets/Resources/"))
                Directory.CreateDirectory("Assets/Resources/");

            PrefabUtility.CreatePrefab(kPrefabTogglePath, rootGO);
#endif
        }

        private static void CreateToggleGroupPrefab()
        {
#if UNITY_EDITOR
            var rootGO = new GameObject("rootGo");
            GameObject canvasGO = ComponentCreator.CreateCanvasRoot("Canvas");
            canvasGO.transform.SetParent(rootGO.transform);

            // factory function alias for brevity
            var c = ComponentCreator.CreationInfo.Creator;
            ComponentCreator.CreationInfo templateGroup =
                c("ToggleGroup") // object name
                    .With<ToggleGroup>();

            for (int i = 0; i < nbToggleInGroup; ++i)
            {
                var t = ComponentCreator.CreationInfo.Creator;
                ComponentCreator.CreationInfo templateToggle =
                    t("TestToggle" + i) // object name
                        .With<Toggle>()
                        .With<ToggleTestImageHook>();
                templateGroup.Children.Add(templateToggle);
            }

            var nameToObjectMapping = new Dictionary<string, GameObject>();
            ComponentCreator.CreateHierarchy(info: templateGroup, parent: canvasGO, dict: nameToObjectMapping);

            for (int i = 0; i < nbToggleInGroup; ++i)
            {
                var toggle = nameToObjectMapping["TestToggle" + i].GetComponent<Toggle>();
                toggle.enabled = true;
                toggle.graphic = nameToObjectMapping["TestToggle" + i].GetComponent<ToggleTestImageHook>();
                toggle.graphic.canvasRenderer.SetColor(Color.white);
            }

            if (!Directory.Exists("Assets/Resources/"))
                Directory.CreateDirectory("Assets/Resources/");

            PrefabUtility.CreatePrefab(kPrefabToggleGroupPath, rootGO);
#endif
        }

        [SetUp]
        public virtual void TestSetup()
        {
            m_PrefabRoot = Object.Instantiate(Resources.Load("TestToggle")) as GameObject;
            var node = m_PrefabRoot.transform.Find("Canvas/TestToggle");
            m_toggle.Add(node.gameObject.GetComponent<Toggle>());
        }

        [TearDown]
        public virtual void TearDown()
        {
            m_toggle.Clear();
            Object.Destroy(m_PrefabRoot);
        }
    }

    [Ignore("Results in error building player (1139182)")]
    class ToggleTests : BaseToggleTests
    {
        [Test]
        public void SetIsOnWithoutNotifyWillNotNotify()
        {
            m_toggle[0].isOn = false;
            bool calledOnValueChanged = false;
            m_toggle[0].onValueChanged.AddListener(b => { calledOnValueChanged = true; });
            m_toggle[0].SetIsOnWithoutNotify(true);
            Assert.IsTrue(m_toggle[0].isOn);
            Assert.IsFalse(calledOnValueChanged);
        }

        [Test]
        public void NonInteractableCantBeToggled()
        {
            m_toggle[0].isOn = true;
            Assert.IsTrue(m_toggle[0].isOn);
            m_toggle[0].interactable = false;
            m_toggle[0].OnSubmit(null);
            Assert.IsTrue(m_toggle[0].isOn);
        }

        [Test]
        public void InactiveCantBeToggled()
        {
            m_toggle[0].isOn = true;
            Assert.IsTrue(m_toggle[0].isOn);
            m_toggle[0].enabled = false;
            m_toggle[0].OnSubmit(null);
            Assert.IsTrue(m_toggle[0].isOn);
        }

        [UnityTest][Ignore("Test doesn't pass")]
        public IEnumerator ToggleOnShouldStartTransition()
        {
            m_toggle[0].toggleTransition = Toggle.ToggleTransition.Fade;
            m_toggle[0].isOn = false;
            m_toggle[0].OnSubmit(null);
            var hook = m_toggle[0].graphic as ToggleTestImageHook;
            yield return new WaitForSeconds(hook.durationTween);
            Assert.AreEqual(1, m_toggle[0].graphic.canvasRenderer.GetColor().a);
        }

        [UnityTest][Ignore("Test doesn't pass")]
        public IEnumerator ToggleOffShouldStartTransition()
        {
            m_toggle[0].toggleTransition = Toggle.ToggleTransition.Fade;
            m_toggle[0].isOn = true;
            m_toggle[0].OnSubmit(null);
            var hook = m_toggle[0].graphic as ToggleTestImageHook;
            yield return new WaitForSeconds(hook.durationTween);
            Assert.AreEqual(0, m_toggle[0].graphic.canvasRenderer.GetColor().a);
        }

        [UnityTest][Ignore("Test doesn't pass")]
        public IEnumerator ToggleOnTransitionNoneShouldNotStartTransition()
        {
            m_toggle[0].toggleTransition = Toggle.ToggleTransition.None;
            m_toggle[0].isOn = false;
            m_toggle[0].OnSubmit(null);
            var hook = m_toggle[0].graphic as ToggleTestImageHook;
            yield return new WaitForSeconds(hook.durationTween);
            Assert.AreEqual(1, m_toggle[0].graphic.canvasRenderer.GetColor().a);
        }

        [UnityTest][Ignore("Test doesn't pass")]
        public IEnumerator ToggleOffTransitionNoneShouldNotStartTransition()
        {
            m_toggle[0].toggleTransition = Toggle.ToggleTransition.None;
            m_toggle[0].isOn = true;
            m_toggle[0].OnSubmit(null);
            var hook = m_toggle[0].graphic as ToggleTestImageHook;
            yield return new WaitForSeconds(hook.durationTween);
            Assert.AreEqual(0, m_toggle[0].graphic.canvasRenderer.GetColor().a);
        }
    }

    [Ignore("Results in error building player (1139182)")]
    class ToggleGroupTests : BaseToggleTests
    {
        private ToggleGroup m_toggleGroup;

        [SetUp]
        public override void TestSetup()
        {
            m_PrefabRoot = Object.Instantiate(Resources.Load("TestToggleGroup")) as GameObject;

            m_toggleGroup = m_PrefabRoot.GetComponentInChildren<ToggleGroup>();
            m_toggle.AddRange(m_PrefabRoot.GetComponentsInChildren<Toggle>());
        }

        [TearDown]
        public override void TearDown()
        {
            m_toggleGroup = null;
            m_toggle.Clear();
            Object.Destroy(m_PrefabRoot);
        }

        [Test]
        public void TogglingOneShouldDisableOthersInGroup()
        {
            m_toggle[0].group = m_toggleGroup;
            m_toggle[1].group = m_toggleGroup;
            m_toggle[0].isOn = true;
            m_toggle[1].isOn = true;
            Assert.IsFalse(m_toggle[0].isOn);
            Assert.IsTrue(m_toggle[1].isOn);
        }

        [Test]
        public void DisallowSwitchOffShouldKeepToggleOnWhenClicking()
        {
            m_toggle[0].group = m_toggleGroup;
            m_toggle[1].group = m_toggleGroup;
            m_toggle[0].isOn = true;
            Assert.IsTrue(m_toggle[0].isOn);
            m_toggle[0].OnPointerClick(new PointerEventData(EventSystem.current) { button = PointerEventData.InputButton.Left });
            Assert.IsTrue(m_toggle[0].isOn);
            Assert.IsFalse(m_toggle[1].isOn);
        }

        [Test]
        public void DisallowSwitchOffShouldDisableToggleWhenClicking()
        {
            m_toggleGroup.allowSwitchOff = true;
            m_toggle[0].group = m_toggleGroup;
            m_toggle[1].group = m_toggleGroup;
            m_toggle[0].isOn = true;
            Assert.IsTrue(m_toggle[0].isOn);
            m_toggle[0].OnPointerClick(new PointerEventData(EventSystem.current) { button = PointerEventData.InputButton.Left });
            Assert.IsFalse(m_toggle[0].isOn);
            Assert.IsFalse(m_toggle[1].isOn);
        }
    }
}
*/
