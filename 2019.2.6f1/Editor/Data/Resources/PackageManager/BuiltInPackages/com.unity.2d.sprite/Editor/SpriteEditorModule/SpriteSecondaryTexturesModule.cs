using System;
using System.Collections.Generic;
using UnityEditor.U2D.Sprites;
using UnityEditorInternal;
using UnityEngine;

namespace UnityEditor._2D.Sprite.Editor
{
    internal class SpriteSecondaryTexturesModule : SpriteEditorModuleBase
    {
        private static class Styles
        {
            public static readonly string invalidEntriesWarning = L10n.Tr("Invalid secondary Texture entries (without names or Textures) have been removed.");
            public static readonly string nameUniquenessWarning = L10n.Tr("Every secondary Texture attached to the Sprite must have a unique name.");
            public static readonly string builtInNameCollisionWarning = L10n.Tr("The names _MainTex and _AlphaTex are reserved for internal use.");
            public static readonly GUIContent panelTitle = EditorGUIUtility.TrTextContent("Secondary Textures");
            public static readonly GUIContent name = EditorGUIUtility.TrTextContent("Name");
            public static readonly GUIContent texture = EditorGUIUtility.TrTextContent("Texture");
            public const float textFieldDropDownWidth = 18.0f;
        }

        TextureImporter m_TextureImporter;
        ReorderableList m_ReorderableList;
        Vector2 m_ReorderableListScrollPosition;
        string[] m_SuggestedNames;

        internal List<SecondarySpriteTexture> secondaryTextureList { get; private set; }

        public override string moduleName
        {
            get { return "Secondary Textures"; }
        }

        public override bool ApplyRevert(bool apply)
        {
            if (apply)
            {
                var textureImporter = new SerializedObject(m_TextureImporter);
                var secondaryTextures = textureImporter.FindProperty("m_SpriteSheet.m_SecondaryTextures");

                // Remove invalid entries.
                List<SecondarySpriteTexture> validEntries = secondaryTextureList.FindAll(x => (x.name != null && x.name != "" && x.texture != null));
                if (validEntries.Count < secondaryTextureList.Count)
                    Debug.Log(Styles.invalidEntriesWarning);

                secondaryTextures.arraySize = validEntries.Count;
                for (int i = 0; i < validEntries.Count; ++i)
                {
                    var e = secondaryTextures.GetArrayElementAtIndex(i);
                    e.FindPropertyRelative("name").stringValue = validEntries[i].name;
                    e.FindPropertyRelative("texture").objectReferenceValue = validEntries[i].texture;
                }

                textureImporter.ApplyModifiedPropertiesWithoutUndo();
            }

            return true;
        }

        public override bool CanBeActivated()
        {
            TextureImporter textureImporter = spriteEditor.GetDataProvider<ISpriteEditorDataProvider>()?.targetObject as TextureImporter;

            if (textureImporter != null)
                return textureImporter.textureType == TextureImporterType.Sprite;
            else
                return false;
        }

        public override void DoMainGUI()
        {
        }

        public override void DoPostGUI()
        {
            if (m_TextureImporter == null)
                return;

            using (new EditorGUI.DisabledScope(spriteEditor.editingDisabled))
            {
                var windowDimension = spriteEditor.windowDimension;
                Rect panelRect = new Rect(windowDimension.width - 300, windowDimension.height - 300, 290, 290);
                GUILayout.BeginArea(panelRect, Styles.panelTitle, GUI.skin.window);
                m_ReorderableListScrollPosition = GUILayout.BeginScrollView(m_ReorderableListScrollPosition);
                m_ReorderableList.DoLayoutList();
                GUILayout.EndScrollView();
                GUILayout.EndArea();

                // Deselect the list item if left click outside the panel area.
                UnityEngine.Event e = UnityEngine.Event.current;
                if (e.type == EventType.MouseDown && e.button == 0 && !panelRect.Contains(e.mousePosition))
                {
                    m_ReorderableList.index = -1;
                    spriteEditor.RequestRepaint();
                }
            }

            // Preview the current selected secondary texture.
            Texture2D previewTexture = null;
            int width = 0, height = 0;

            var textureDataProvider = spriteEditor.GetDataProvider<ITextureDataProvider>();
            if (textureDataProvider != null)
            {
                previewTexture = textureDataProvider.previewTexture;
                textureDataProvider.GetTextureActualWidthAndHeight(out width, out height);
            }

            if (m_ReorderableList.index >= 0 && m_ReorderableList.index < secondaryTextureList.Count)
                previewTexture = secondaryTextureList[m_ReorderableList.index].texture;

            if (previewTexture != null)
                spriteEditor.SetPreviewTexture(previewTexture, width, height);
        }

        public override void DoToolbarGUI(Rect drawArea)
        {
        }

        public override void OnModuleActivate()
        {
            m_TextureImporter = spriteEditor.GetDataProvider<ISpriteEditorDataProvider>().targetObject as TextureImporter;
            if (m_TextureImporter == null)
                return;

            var textureImporter = new SerializedObject(m_TextureImporter);
            var secondaryTextures = textureImporter.FindProperty("m_SpriteSheet.m_SecondaryTextures");

            secondaryTextureList = new List<SecondarySpriteTexture>(secondaryTextures.arraySize);
            for (int i = 0; i < secondaryTextures.arraySize; ++i)
            {
                var e = secondaryTextures.GetArrayElementAtIndex(i);
                secondaryTextureList.Add(new SecondarySpriteTexture()
                {
                    name = e.FindPropertyRelative("name").stringValue,
                    texture = e.FindPropertyRelative("texture").objectReferenceValue as Texture2D,
                });
            }

            m_ReorderableListScrollPosition = Vector2.zero;
            m_ReorderableList = new ReorderableList(secondaryTextureList, typeof(SecondarySpriteTexture), false, false, true, true);
            m_ReorderableList.drawElementCallback = DrawSpriteSecondaryTextureElement;
            m_ReorderableList.onAddCallback = AddSpriteSecondaryTextureElement;
            m_ReorderableList.onRemoveCallback = RemoveSpriteSecondaryTextureElement;
            m_ReorderableList.onCanAddCallback = CanAddSpriteSecondaryTextureElement;
            m_ReorderableList.elementHeightCallback = (int index) => (EditorGUIUtility.singleLineHeight * 3) + 5;

            spriteEditor.selectedSpriteRect = null;

            string suggestedNamesPrefs = EditorPrefs.GetString("SecondarySpriteTexturePropertyNames");
            if (!string.IsNullOrEmpty(suggestedNamesPrefs))
            {
                m_SuggestedNames = suggestedNamesPrefs.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);

                for (int i = 0; i < m_SuggestedNames.Length; ++i)
                    m_SuggestedNames[i] = m_SuggestedNames[i].Trim();

                Array.Sort(m_SuggestedNames);
            }
            else
                m_SuggestedNames = null;
        }

        public override void OnModuleDeactivate()
        {
            // Reset to display the main texture.
            ITextureDataProvider textureDataProvider = spriteEditor.GetDataProvider<ITextureDataProvider>();
            if (textureDataProvider != null && textureDataProvider.previewTexture != null)
            {
                Texture2D mainTexture = textureDataProvider.previewTexture;
                int width = 0, height = 0;
                textureDataProvider.GetTextureActualWidthAndHeight(out width, out height);
                spriteEditor.SetPreviewTexture(mainTexture, width, height);
            }
        }

        void DrawSpriteSecondaryTextureElement(Rect rect, int index, bool isActive, bool isFocused)
        {
            bool dataModified = false;
            float oldLabelWidth = EditorGUIUtility.labelWidth;
            EditorGUIUtility.labelWidth = 70.0f;
            SecondarySpriteTexture secondaryTexture = secondaryTextureList[index];

            // "Name" text field
            EditorGUI.BeginChangeCheck();
            var r = new Rect(rect.x, rect.y + 5, rect.width - Styles.textFieldDropDownWidth, EditorGUIUtility.singleLineHeight);
            string newName = EditorGUI.DelayedTextField(r, Styles.name, secondaryTexture.name);
            dataModified = EditorGUI.EndChangeCheck();

            // Suggested names
            if (m_SuggestedNames != null)
            {
                var popupRect = new Rect(r.x + r.width, r.y, Styles.textFieldDropDownWidth, EditorGUIUtility.singleLineHeight);
                EditorGUI.BeginChangeCheck();
                int selected = EditorGUI.Popup(popupRect, -1, m_SuggestedNames, EditorStyles.textFieldDropDown);
                if (EditorGUI.EndChangeCheck())
                {
                    newName = m_SuggestedNames[selected];
                    dataModified = true;
                }
            }

            if (dataModified)
            {
                if (!string.IsNullOrEmpty(newName) && newName != secondaryTexture.name && secondaryTextureList.Exists(x => x.name == newName))
                    Debug.LogWarning(Styles.nameUniquenessWarning);
                else if (newName == "_MainTex" || newName == "_AlphaTex")
                    Debug.LogWarning(Styles.builtInNameCollisionWarning);
                else
                    secondaryTexture.name = newName;
            }

            // "Texture" object field
            EditorGUI.BeginChangeCheck();
            r.width = rect.width;
            r.y += EditorGUIUtility.singleLineHeight;
            secondaryTexture.texture = EditorGUI.ObjectField(r, Styles.texture, secondaryTexture.texture, typeof(Texture2D), false) as Texture2D;
            dataModified = dataModified || EditorGUI.EndChangeCheck();

            if (dataModified)
            {
                secondaryTextureList[index] = secondaryTexture;
                spriteEditor.SetDataModified();
            }

            EditorGUIUtility.labelWidth = oldLabelWidth;
        }

        void AddSpriteSecondaryTextureElement(ReorderableList list)
        {
            m_ReorderableListScrollPosition += new Vector2(0.0f, list.elementHeightCallback(0));
            secondaryTextureList.Add(new SecondarySpriteTexture());
            spriteEditor.SetDataModified();
        }

        void RemoveSpriteSecondaryTextureElement(ReorderableList list)
        {
            secondaryTextureList.RemoveAt(list.index);
            spriteEditor.SetDataModified();
        }

        bool CanAddSpriteSecondaryTextureElement(ReorderableList list)
        {
            return list.count < 8;
        }
    }
}
