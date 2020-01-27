import java.util.HashMap;

public class LRUCache {
    HashMap<Integer, DNode> hm = new HashMap<>();
    DNode head, tail;

    int maxSize;
    int elements;

    public LRUCache(int capacity) {
        this.maxSize = capacity;
        elements = 0;
        head = new DNode();
        tail = new DNode();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        if(hm.containsKey(key)){
            DNode res = hm.get(key);
            moveToFront(res);
            return res.value;
        } else {
            return -1;
        }
    }

    public void put(int key, int value) {
        if(hm.containsKey(key)){
            DNode res = hm.get(key);
            moveToFront(res);
            res.value = value;
        } else {
            DNode res = new DNode();
            res.key = key;
            res.value = value;
            hm.put(key, res);
            addToFront(res);
            if(elements < maxSize){
                elements++;
            } else {
                deleteLast();
            }
        }
    }

    public void moveToFront(DNode node){
        DNode previous = node.prev;
        DNode actualnext = node.next;
        previous.next = actualnext;
        actualnext.prev = previous;

        addToFront(node);
    }

    public void addToFront(DNode node){
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    public void deleteLast(){
        int key = tail.prev.key;
        hm.remove(key);
        DNode last = tail.prev.prev;
        last.next = tail;
        tail.prev = last;
    }


    private class DNode {
        int key;
        int value;

        DNode prev;
        DNode next;
    }
}
