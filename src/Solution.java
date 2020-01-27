import java.util.*;

public class Solution {
    static List<List<Integer>> res = new ArrayList<>();
    public static List<List<Integer>> levelOrder(TreeNode root) {
        if(root == null) return res;
        helper(root, 0);
        return res;
    }

    public static void helper(TreeNode root, int level){
        if(level == res.size()){
            res.add(new ArrayList<>());
        }
        res.get(level).add(root.val);
        if(root.left != null){
            helper(root.right, level + 1);
        }
        if(root.right != null){
            helper(root.left, level + 1);
        }
    }
}
