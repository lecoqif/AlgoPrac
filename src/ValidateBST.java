public class ValidateBST {
    public boolean isValidBST(TreeNode root) {
        if(root == null) return true;
        return helper(root.left, root.val, null) && helper(root.right, null, root.val);
    }

    public boolean helper(TreeNode node, Integer upper, Integer lower){
        if(node == null) return true;
        if(upper != null && node.val > upper) return false;
        if(lower != null && node.val < lower) return false;
        return helper(node.left, node.val, lower) && helper(node.right, upper, node.val);
    }
}
