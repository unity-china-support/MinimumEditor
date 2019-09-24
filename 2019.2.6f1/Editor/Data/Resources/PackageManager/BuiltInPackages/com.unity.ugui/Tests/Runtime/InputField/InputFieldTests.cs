using System;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.TestTools;
using System.Collections;
using System.IO;
using UnityEditor;
using UnityEngine.UI;
using System.Reflection;

namespace InputfieldTests
{
    public abstract class BaseInputFieldTests : IPrebuildSetup
    {
        protected GameObject m_PrefabRoot;

        protected const string kPrefabPath = "Assets/Resources/InputFieldPrefab.prefab";

        public void Setup()
        {
#if UNITY_EDITOR
            var rootGO = new GameObject("rootGo");

            var canvasGO = new GameObject("Canvas", typeof(Canvas));
            canvasGO.transform.SetParent(rootGO.transform);
            var canvas = canvasGO.GetComponent<Canvas>();
            canvas.referencePixelsPerUnit = 100;

            GameObject inputFieldGO = new GameObject("InputField", typeof(RectTransform), typeof(InputField));
            inputFieldGO.transform.SetParent(canvasGO.transform);

            GameObject textGO = new GameObject("Text", typeof(RectTransform), typeof(Text));
            textGO.transform.SetParent(inputFieldGO.transform);

            GameObject eventSystemGO = new GameObject("EventSystem", typeof(EventSystem), typeof(FakeInputModule));
            eventSystemGO.transform.SetParent(rootGO.transform);

            InputField inputField = inputFieldGO.GetComponent<InputField>();

            inputField.interactable = true;
            inputField.enabled = true;
            inputField.textComponent = textGO.GetComponent<Text>();
            inputField.textComponent.fontSize = 12;
            inputField.textComponent.supportRichText = false;

            if (!Directory.Exists("Assets/Resources/"))
                Directory.CreateDirectory("Assets/Resources/");

            PrefabUtility.SaveAsPrefabAsset(rootGO, kPrefabPath);
            GameObject.DestroyImmediate(rootGO);
#endif
        }

        [SetUp]
        public virtual void TestSetup()
        {
            m_PrefabRoot = UnityEngine.Object.Instantiate(Resources.Load("InputFieldPrefab")) as GameObject;

            FieldInfo inputModule = typeof(EventSystem).GetField("m_CurrentInputModule", BindingFlags.NonPublic | BindingFlags.Instance);
            inputModule.SetValue(m_PrefabRoot.GetComponentInChildren<EventSystem>(), m_PrefabRoot.GetComponentInChildren<FakeInputModule>());
        }

        [TearDown]
        public virtual void TearDown()
        {
            FontUpdateTracker.UntrackText(m_PrefabRoot.GetComponentInChildren<Text>());
            GameObject.Destroy(m_PrefabRoot);
        }
    }

    // Tests we don't need to run for both types of keyboard
    public class GenericInputFieldTests : BaseInputFieldTests
    {
        [UnityTest]
        public IEnumerator CannotFocusIfNotTextComponent()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            BaseEventData eventData = new BaseEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            inputField.textComponent = null;

            inputField.OnSelect(eventData);
            yield return null;

            Assert.False(inputField.isFocused);
        }

        [UnityTest]
        public IEnumerator CannotFocusIfNullFont()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            BaseEventData eventData = new BaseEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            inputField.textComponent.font = null;

            inputField.OnSelect(eventData);
            yield return null;

            Assert.False(inputField.isFocused);
        }

        [UnityTest]
        public IEnumerator CannotFocusIfNotActive()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            BaseEventData eventData = new BaseEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            inputField.enabled = false;

            inputField.OnSelect(eventData);
            yield return null;

            Assert.False(inputField.isFocused);
        }

        [UnityTest]
        public IEnumerator CannotFocusWithoutEventSystem()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            UnityEngine.Object.DestroyImmediate(m_PrefabRoot.GetComponentInChildren<FakeInputModule>());

            yield return null;

            UnityEngine.Object.DestroyImmediate(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            BaseEventData eventData = new BaseEventData(null);

            yield return null;

            inputField.OnSelect(eventData);
            yield return null;

            Assert.False(inputField.isFocused);
        }

        [Test]
        public void FocusesOnSelect()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            BaseEventData eventData = new BaseEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            inputField.OnSelect(eventData);

            MethodInfo lateUpdate = typeof(InputField).GetMethod("LateUpdate", BindingFlags.NonPublic | BindingFlags.Instance);
            lateUpdate.Invoke(inputField, null);

            Assert.True(inputField.isFocused);
        }

        [Test]
        public void InputFieldSetTextWithoutNotifyWillNotNotify()
        {
            InputField i = m_PrefabRoot.GetComponentInChildren<InputField>();
            i.text = "Hello";

            bool calledOnValueChanged = false;

            i.onValueChanged.AddListener(s => { calledOnValueChanged = true; });

            i.SetTextWithoutNotify("Goodbye");

            Assert.IsTrue(i.text == "Goodbye");
            Assert.IsFalse(calledOnValueChanged);
        }

        [Test]
        public void ContentTypeSetsValues()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            inputField.contentType = InputField.ContentType.Standard;
            Assert.AreEqual(InputField.InputType.Standard, inputField.inputType);
            Assert.AreEqual(TouchScreenKeyboardType.Default, inputField.keyboardType);
            Assert.AreEqual(InputField.CharacterValidation.None, inputField.characterValidation);

            inputField.contentType = InputField.ContentType.Autocorrected;
            Assert.AreEqual(InputField.InputType.AutoCorrect, inputField.inputType);
            Assert.AreEqual(TouchScreenKeyboardType.Default, inputField.keyboardType);
            Assert.AreEqual(InputField.CharacterValidation.None, inputField.characterValidation);

            inputField.contentType = InputField.ContentType.IntegerNumber;
            Assert.AreEqual(InputField.LineType.SingleLine, inputField.lineType);
            Assert.AreEqual(InputField.InputType.Standard, inputField.inputType);
            Assert.AreEqual(TouchScreenKeyboardType.NumberPad, inputField.keyboardType);
            Assert.AreEqual(InputField.CharacterValidation.Integer, inputField.characterValidation);

            inputField.contentType = InputField.ContentType.DecimalNumber;
            Assert.AreEqual(InputField.LineType.SingleLine, inputField.lineType);
            Assert.AreEqual(InputField.InputType.Standard, inputField.inputType);
            Assert.AreEqual(TouchScreenKeyboardType.NumbersAndPunctuation, inputField.keyboardType);
            Assert.AreEqual(InputField.CharacterValidation.Decimal, inputField.characterValidation);

            inputField.contentType = InputField.ContentType.Alphanumeric;
            Assert.AreEqual(InputField.LineType.SingleLine, inputField.lineType);
            Assert.AreEqual(InputField.InputType.Standard, inputField.inputType);
            Assert.AreEqual(TouchScreenKeyboardType.ASCIICapable, inputField.keyboardType);
            Assert.AreEqual(InputField.CharacterValidation.Alphanumeric, inputField.characterValidation);

            inputField.contentType = InputField.ContentType.Name;
            Assert.AreEqual(InputField.LineType.SingleLine, inputField.lineType);
            Assert.AreEqual(InputField.InputType.Standard, inputField.inputType);
            Assert.AreEqual(TouchScreenKeyboardType.NamePhonePad, inputField.keyboardType);
            Assert.AreEqual(InputField.CharacterValidation.Name, inputField.characterValidation);

            inputField.contentType = InputField.ContentType.EmailAddress;
            Assert.AreEqual(InputField.LineType.SingleLine, inputField.lineType);
            Assert.AreEqual(InputField.InputType.Standard, inputField.inputType);
            Assert.AreEqual(TouchScreenKeyboardType.EmailAddress, inputField.keyboardType);
            Assert.AreEqual(InputField.CharacterValidation.EmailAddress, inputField.characterValidation);

            inputField.contentType = InputField.ContentType.Password;
            Assert.AreEqual(InputField.LineType.SingleLine, inputField.lineType);
            Assert.AreEqual(InputField.InputType.Password, inputField.inputType);
            Assert.AreEqual(TouchScreenKeyboardType.Default, inputField.keyboardType);
            Assert.AreEqual(InputField.CharacterValidation.None, inputField.characterValidation);

            inputField.contentType = InputField.ContentType.Pin;
            Assert.AreEqual(InputField.LineType.SingleLine, inputField.lineType);
            Assert.AreEqual(InputField.InputType.Password, inputField.inputType);
            Assert.AreEqual(TouchScreenKeyboardType.NumberPad, inputField.keyboardType);
            Assert.AreEqual(InputField.CharacterValidation.Integer, inputField.characterValidation);
        }

        [Test]
        public void SettingLineTypeDoesNotChangesContentTypeToCustom([Values(InputField.ContentType.Standard, InputField.ContentType.Autocorrected)] InputField.ContentType type)
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            inputField.contentType = type;

            inputField.lineType = InputField.LineType.MultiLineNewline;

            Assert.AreEqual(type, inputField.contentType);
        }

        [Test]
        public void SettingLineTypeChangesContentTypeToCustom()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            inputField.contentType = InputField.ContentType.Name;

            inputField.lineType = InputField.LineType.MultiLineNewline;

            Assert.AreEqual(InputField.ContentType.Custom, inputField.contentType);
        }

        [Test]
        public void SettingInputChangesContentTypeToCustom()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            inputField.contentType = InputField.ContentType.Name;

            inputField.inputType = InputField.InputType.Password;

            Assert.AreEqual(InputField.ContentType.Custom, inputField.contentType);
        }

        [Test]
        public void SettingCharacterValidationChangesContentTypeToCustom()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            inputField.contentType = InputField.ContentType.Name;

            inputField.characterValidation = InputField.CharacterValidation.None;

            Assert.AreEqual(InputField.ContentType.Custom, inputField.contentType);
        }

        [Test]
        public void SettingKeyboardTypeChangesContentTypeToCustom()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            inputField.contentType = InputField.ContentType.Name;

            inputField.keyboardType = TouchScreenKeyboardType.ASCIICapable;

            Assert.AreEqual(InputField.ContentType.Custom, inputField.contentType);
        }

        [UnityTest]
        public IEnumerator CaretRectSameSizeAsTextRect()
        {
            InputField inputfield = m_PrefabRoot.GetComponentInChildren<InputField>();
            HorizontalLayoutGroup lg = inputfield.gameObject.AddComponent<HorizontalLayoutGroup>();
            lg.childControlWidth = true;
            lg.childControlHeight = false;
            lg.childForceExpandWidth = true;
            lg.childForceExpandHeight = true;
            ContentSizeFitter csf = inputfield.gameObject.AddComponent<ContentSizeFitter>();
            csf.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
            csf.verticalFit = ContentSizeFitter.FitMode.Unconstrained;
            inputfield.text = "Hello World!";

            yield return new WaitForSeconds(1.0f);

            Rect prevTextRect = inputfield.textComponent.rectTransform.rect;
            Rect prevCaretRect = (inputfield.textComponent.transform.parent.GetChild(0) as RectTransform).rect;
            inputfield.text = "Hello World!Hello World!Hello World!";

            LayoutRebuilder.MarkLayoutForRebuild(inputfield.transform as RectTransform);

            yield return new WaitForSeconds(1.0f);

            Rect newTextRect = inputfield.textComponent.rectTransform.rect;
            Rect newCaretRect = (inputfield.textComponent.transform.parent.GetChild(0) as RectTransform).rect;

            Assert.IsFalse(prevTextRect == newTextRect);
            Assert.IsTrue(prevTextRect == prevCaretRect);
            Assert.IsFalse(prevCaretRect == newCaretRect);
            Assert.IsTrue(newTextRect == newCaretRect);
        }
    }

    [TestFixture]
    public class DesktopInputFieldTests : BaseInputFieldTests
    {
        [SetUp]
        public override void TestSetup()
        {
            base.TestSetup();
        }

        public override void TearDown()
        {
            GUIUtility.systemCopyBuffer = null;
            base.TearDown();
        }

        [Test]
        public void FocusOnPointerClickWithLeftButton()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            PointerEventData data = new PointerEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            data.button = PointerEventData.InputButton.Left;
            inputField.OnPointerClick(data);

            MethodInfo lateUpdate = typeof(InputField).GetMethod("LateUpdate", BindingFlags.NonPublic | BindingFlags.Instance);
            lateUpdate.Invoke(inputField, null);

            Assert.IsTrue(inputField.isFocused);
        }

        [UnityTest]
        public IEnumerator DoesNotFocusOnPointerClickWithRightOrMiddleButton()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            PointerEventData data = new PointerEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            data.button = PointerEventData.InputButton.Middle;
            inputField.OnPointerClick(data);
            yield return null;

            data.button = PointerEventData.InputButton.Right;
            inputField.OnPointerClick(data);
            yield return null;

            Assert.IsFalse(inputField.isFocused);
        }
    }

    [TestFixture]
    [UnityPlatform(exclude = new RuntimePlatform[]
    {
        RuntimePlatform.Android /* case 1094042 */
    })]
    public class TouchInputFieldTests : BaseInputFieldTests
    {
        protected const string kDefaultInputStr = "foobar";

        const string kEmailSpecialCharacters = "!#$%&'*+-/=?^_`{|}~";

        public struct CharValidationTestData
        {
            public string input, output;
            public InputField.CharacterValidation validation;

            public CharValidationTestData(string input, string output, InputField.CharacterValidation validation)
            {
                this.input = input;
                this.output = output;
                this.validation = validation;
            }

            public override string ToString()
            {
                // these won't properly show up if test runners UI if we don't replace it
                string input = this.input.Replace(kEmailSpecialCharacters, "specialchars");
                string output = this.output.Replace(kEmailSpecialCharacters, "specialchars");
                return string.Format("input={0}, output={1}, validation={2}", input, output, validation);
            }
        }

        [Test]
        [TestCase("*Azé09", "*Azé09", InputField.CharacterValidation.None)]
        [TestCase("*Azé09?.", "Az09", InputField.CharacterValidation.Alphanumeric)]
        [TestCase("Abc10x", "10", InputField.CharacterValidation.Integer)]
        [TestCase("-10", "-10", InputField.CharacterValidation.Integer)]
        [TestCase("10.0", "100", InputField.CharacterValidation.Integer)]
        [TestCase("10.0", "10.0", InputField.CharacterValidation.Decimal)]
        [TestCase(" -10.0x", "-10.0", InputField.CharacterValidation.Decimal)]
        [TestCase("10,0", "10,0", InputField.CharacterValidation.Decimal)]
        [TestCase(" -10,0x", "-10,0", InputField.CharacterValidation.Decimal)]
        [TestCase("A10,0 ", "10,0", InputField.CharacterValidation.Decimal)]
        [TestCase("A'a aaa  aaa", "A'a Aaa Aaa", InputField.CharacterValidation.Name)]
        [TestCase(" _JOHN*   (Doe)", "John Doe", InputField.CharacterValidation.Name)]
        [TestCase("johndoe@unity3d.com", "johndoe@unity3d.com", InputField.CharacterValidation.EmailAddress)]
        [TestCase(">john doe\\@unity3d.com", "johndoe@unity3d.com", InputField.CharacterValidation.EmailAddress)]
        [TestCase(kEmailSpecialCharacters + "@unity3d.com", kEmailSpecialCharacters + "@unity3d.com", InputField.CharacterValidation.EmailAddress)]
        public void HonorsCharacterValidationSettingsAssignment(string input, string output, InputField.CharacterValidation validation)
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            inputField.characterValidation = validation;
            inputField.text = input;
            Assert.AreEqual(output, inputField.text, string.Format("Failed character validation: input ={0}, output ={1}, validation ={2}",
                input.Replace(kEmailSpecialCharacters, "specialchars"),
                output.Replace(kEmailSpecialCharacters, "specialchars"),
                validation));
        }

        [UnityTest]
        [TestCase("*Azé09", "*Azé09", InputField.CharacterValidation.None, ExpectedResult = null)]
        [TestCase("*Azé09?.", "Az09", InputField.CharacterValidation.Alphanumeric, ExpectedResult = null)]
        [TestCase("Abc10x", "10", InputField.CharacterValidation.Integer, ExpectedResult = null)]
        [TestCase("-10", "-10", InputField.CharacterValidation.Integer, ExpectedResult = null)]
        [TestCase("10.0", "100", InputField.CharacterValidation.Integer, ExpectedResult = null)]
        [TestCase("10.0", "10.0", InputField.CharacterValidation.Decimal, ExpectedResult = null)]
        [TestCase(" -10.0x", "-10.0", InputField.CharacterValidation.Decimal, ExpectedResult = null)]
        [TestCase("10,0", "10,0", InputField.CharacterValidation.Decimal, ExpectedResult = null)]
        [TestCase(" -10,0x", "-10,0", InputField.CharacterValidation.Decimal, ExpectedResult = null)]
        [TestCase("A10,0 ", "10,0", InputField.CharacterValidation.Decimal, ExpectedResult = null)]
        [TestCase("A'a aaa  aaa", "A'a Aaa Aaa", InputField.CharacterValidation.Name, ExpectedResult = null)]
        [TestCase(" _JOHN*   (Doe)", "John Doe", InputField.CharacterValidation.Name, ExpectedResult = null)]
        [TestCase("johndoe@unity3d.com", "johndoe@unity3d.com", InputField.CharacterValidation.EmailAddress, ExpectedResult = null)]
        [TestCase(">john doe\\@unity3d.com", "johndoe@unity3d.com", InputField.CharacterValidation.EmailAddress, ExpectedResult = null)]
        [TestCase(kEmailSpecialCharacters + "@unity3d.com", kEmailSpecialCharacters + "@unity3d.com", InputField.CharacterValidation.EmailAddress, ExpectedResult = null)]
        public IEnumerator HonorsCharacterValidationSettingsTypingWithSelection(string input, string output, InputField.CharacterValidation validation)
        {
            if (!TouchScreenKeyboard.isSupported)
                yield break;
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            BaseEventData eventData = new BaseEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            inputField.characterValidation = validation;
            inputField.text = input;

            inputField.OnSelect(eventData);
            yield return null;

            Assert.AreEqual(output, inputField.text, string.Format("Failed character validation: input ={0}, output ={1}, validation ={2}",
                input.Replace(kEmailSpecialCharacters, "specialchars"),
                output.Replace(kEmailSpecialCharacters, "specialchars"),
                validation));
        }

        [Test]
        public void AssignmentAgainstCharacterLimit([Values("ABC", "abcdefghijkl")] string text)
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            // test assignment
            inputField.characterLimit = 5;
            inputField.text = text;
            Assert.AreEqual(text.Substring(0, Math.Min(text.Length, inputField.characterLimit)), inputField.text);
        }

        [Test] // regression test 793119
        public void AssignmentAgainstCharacterLimitWithContentType([Values("Abc", "Abcdefghijkl")] string text)
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            // test assignment
            inputField.characterLimit = 5;
            inputField.contentType = InputField.ContentType.Name;
            inputField.text = text;
            Assert.AreEqual(text.Substring(0, Math.Min(text.Length, inputField.characterLimit)), inputField.text);
        }

        [UnityTest]
        public IEnumerator SendsEndEditEventOnDeselect()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            BaseEventData eventData = new BaseEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            inputField.OnSelect(eventData);
            yield return null;
            var called = false;
            inputField.onEndEdit.AddListener((s) => { called = true; });

            inputField.OnDeselect(eventData);

            Assert.IsTrue(called, "Expected invocation of onEndEdit");
        }

        [Test]
        public void StripsNullCharacters2()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            inputField.text = "a\0b";
            Assert.AreEqual("ab", inputField.text, "\\0 characters should be stripped");
        }

        public override void TearDown()
        {
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            TouchScreenKeyboard.hideInput = false;
            base.TearDown();
        }

        [UnityTest]
        public IEnumerator FocusOpensTouchScreenKeyboard()
        {
            if (!TouchScreenKeyboard.isSupported)
                yield break;
            InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
            BaseEventData eventData = new BaseEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());
            inputField.OnSelect(eventData);
            yield return null;

            Assert.NotNull(inputField.touchScreenKeyboard, "Expect a keyboard to be opened");
        }

        [UnityTest]
        public IEnumerator AssignsShouldHideInput()
        {
            if (Application.platform == RuntimePlatform.Android || Application.platform == RuntimePlatform.IPhonePlayer)
            {
                InputField inputField = m_PrefabRoot.GetComponentInChildren<InputField>();
                BaseEventData eventData = new BaseEventData(m_PrefabRoot.GetComponentInChildren<EventSystem>());

                inputField.shouldHideMobileInput = false;

                inputField.OnSelect(eventData);
                yield return null;

                Assert.IsFalse(inputField.shouldHideMobileInput);
                Assert.IsFalse(TouchScreenKeyboard.hideInput, "Expect TouchScreenKeyboard.hideInput to be set");
            }
        }
    }
}
