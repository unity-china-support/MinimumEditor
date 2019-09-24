using UnityEditor.Presets;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Utility class for creating Tiles
    /// </summary>
    public class TileUtility
    {
        [MenuItem("Assets/Create/Tile", priority = 357)]
        internal static void CreateNewTile()
        {
            string message = string.Format("Save tile'{0}':", "tile");
            string newAssetPath = EditorUtility.SaveFilePanelInProject("Save tile", "New Tile", "asset", message, ProjectWindowUtil.GetActiveFolderPath());

            // If user canceled or save path is invalid, we can't create the tile
            if (string.IsNullOrEmpty(newAssetPath))
                return;

            AssetDatabase.CreateAsset(CreateDefaultTile(), newAssetPath);
        }

        /// <summary>Creates a Tile with defaults based on the Tile preset</summary>
        /// <returns>A Tile with defaults based on the Tile preset</returns>
        public static Tile CreateDefaultTile()
        {
            var tile = ScriptableObject.CreateInstance<Tile>();
            var preset = Preset.GetDefaultForObject(tile);
            if (preset != null)
                preset.ApplyTo(tile);
            return tile;
        }

        internal TileBase CreateTileFromPalette(Sprite sprite)
        {
            TileBase tile = null;
            return tile;
        }

        /// <summary>Creates a Tile with defaults based on the Tile preset and a Sprite set</summary>
        /// <param name="sprite">A Sprite to set the Tile with</param>
        /// <returns>A Tile with defaults based on the Tile preset and a Sprite set</returns>
        [CreateTileFromPalette]
        public static TileBase DefaultTile(Sprite sprite)
        {
            Tile tile = CreateDefaultTile();
            tile.name = sprite.name;
            tile.sprite = sprite;
            tile.color = Color.white;
            return tile;
        }
    }
}
