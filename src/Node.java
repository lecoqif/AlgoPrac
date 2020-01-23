import java.util.*;

public class Node {
    public Character val;
    public Map<Node, Node> children;
    public Node() {}
    public Node(Character _val){
        val = _val;
        children = new HashMap<>();
    }
    public void addChild(Node a){
        children.put(a, a);
    }
}
