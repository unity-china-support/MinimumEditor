using UnityEngine;
using UnityEngine.EventSystems;

public class DragCallbackCheck : MonoBehaviour, IBeginDragHandler, IDragHandler, IEndDragHandler
{
    private bool loggedOnDrag = false;

    public void OnBeginDrag(PointerEventData eventData)
    {
        Debug.Log("OnBeginDrag");
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (loggedOnDrag)
            return;

        loggedOnDrag = true;
        Debug.Log("OnDrag");
    }

    public void OnEndDrag(PointerEventData eventData)
    {
        Debug.Log("OnEndDrag");
    }
}
