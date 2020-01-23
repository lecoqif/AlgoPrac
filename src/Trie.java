
public class Trie {
    private Node root;

    public Trie() {
        root = new Node('/');
    }

    public void insert(String word){
        if(word == null) return;
        int len = word.length();
        int i = 0;
        Node curr = root;
        while(i < len){
            Character a = word.charAt(i);
            Node next = new Node(a);
            if(!curr.children.containsKey(next)){
                curr.addChild(next);
                curr = next;
            } else {
                Node obj = curr.children.get(next);
                curr = obj;
            }
            i++;

        }
    }
//
//    public boolean search(String word){
//
//    }
////
////    public boolean startsWith(String prefix){
////
////    }
}
