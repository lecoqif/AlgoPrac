import java.lang.reflect.Parameter;
import java.lang.reflect.Type;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;
import java.lang.Object.*;

public class Practice {
    public static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hm = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if(hm.containsKey(target - nums[i])) return new int[]{hm.get(target - nums[i]), i};
            else{
                hm.put(nums[i], i);
            }
        }
        return new int[]{};
    }

    public static String mostCommonWord(String paragraph, String[] banned) {
        HashMap<String, Integer> hm = new HashMap<>();
        String[] words = paragraph.split("\\W+");
        Set<String> hashset = new HashSet<>(Arrays.asList(banned));
        for(String word : words){
            word = word.toLowerCase();
            if(hm.containsKey(word)){
                int tmp = hm.get(word);
                hm.put(word, tmp + 1);
            } else {
                hm.put(word, 1);
            }
        }
        String res = "";
        int cnt = 0;
        for(Map.Entry<String, Integer> entry : hm.entrySet()){
            if(entry.getValue() > cnt && !hashset.contains(entry.getKey())){
                res = entry.getKey();
                cnt = entry.getValue();
            }
        }
        return res;
    }

    public static boolean isValid(String s){
        if(s.length() == 0) return true;
        Stack<Character> stack = new Stack<>();
        for(int i = 0; i < s.length(); i++){
            char tmp = s.charAt(i);
            if(tmp == '(' || tmp == '{' || tmp == '['){
                stack.push(s.charAt(i));
            } else {
                if(stack.isEmpty()) return false;
                char popped = stack.pop();
                if(tmp == '}' && popped != '{') return false;
                if(tmp == ']' && popped != '[') return false;
                if(tmp == ')' && popped != '(') return false;
            }
        }
        if(!stack.isEmpty()) return false;
        return true;
    }

    public static int firstUniqChar(String s){
        HashMap<Character, Integer> hm = new HashMap<>();
        s = s.toLowerCase();
        for(int i = 0; i < s.length(); i++){
            char tmp = s.charAt(i);
            hm.put(tmp, hm.getOrDefault(tmp, 0) + 1);
        }
        for(int i = 0; i < s.length(); i++){
            char tmp = s.charAt(i);
            if(hm.get(tmp) == 1) return i;
        }
        return -1;
    }

    public static int missingNumber(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for(int i : nums){
            set.add(i);
        }
        int i = 0;
        while(true){
           if(set.contains(i)){
               i++;
           } else {
               return i;
           }
        }
    }

    public static int betterMissingNumber(int[] nums){
        int len = nums.length;
        int expectedSum = (len + 1)*(len) / 2;
        int actualSum = 0;
        for(int i : nums) actualSum += i;
        return expectedSum - actualSum;
    }

    public static class ListNode {
        int val;
        ListNode next;
        ListNode(int x){ val = x;}
    }

    public static ListNode mergeTwoLists(ListNode l1, ListNode l2){
        ListNode head = new ListNode(0), p = l1, q = l2, curr = head;
        while(p != null && q != null){
            if(p.val <= q.val){
                curr.next = p;
                p = p.next;
            } else {
                curr.next = q;
                q = q.next;
            }
            curr = curr.next;
        }
        curr.next = p == null ? q : p;
        return head.next;
    }

    public static class aNode {
        int val;
        Node next;
        Node random;

        public aNode(int val){
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x){ val = x;}
    }

    public static boolean isSymmetric(TreeNode root){
        if(root == null) return true;
        return dfs_isSym(root.left, root.right);
    }

    public static boolean dfs_isSym(TreeNode a, TreeNode b){
        if(a == null && b == null) return true;
        if(a == null && b != null || a != null && b == null) return false;
        if(a.val != b.val) return false;
        return dfs_isSym(a.left, b.right) && dfs_isSym(a.right, b.left);
    }

    public static int[] twoSum_sorted(int[] numbers, int target) {
        int left = 0;
        int right = numbers.length - 1;
        while(left < right){
            if(numbers[left] + numbers[right] == target) return new int[] {left, right};
            if(numbers[left] + numbers[right] < target) left++;
            if(numbers[left] + numbers[right] > target) right--;
        }
        return new int[]{};
    }

    public static int[][] kClosest(int[][] points, int K) {
        if(points.length == K) return points;
        PriorityQueue<Double> maxHeap = new PriorityQueue<>(points.length);
        HashMap<Double, ArrayList<int[]>> hm = new HashMap<>();
        for(int i = 0; i < points.length; i++){
            double dist = Math.sqrt(Math.pow(points[i][0], 2) + Math.pow(points[i][1], 2));
            maxHeap.add(dist);
            if(hm.containsKey(dist)){
                ArrayList<int[]> temp = hm.get(dist);
                temp.add(points[i]);
                hm.put(dist, temp);
            } else {
                ArrayList<int[]> temp = new ArrayList<>();
                temp.add(points[i]);
                hm.put(dist, temp);
            }
        }
        int i = 0;
        int[][] res = new int[K][];
        while(i < K){
            double dist = maxHeap.poll();
            ArrayList<int[]> temp = hm.get(dist);
            for(int k = 0; k < temp.size(); k++){
                res[i] = temp.get(k);
                i++;
            }
        }
        return res;
    }

    public static class Islands {
        char[][] grid;

        public int numIslands(char[][] grid1) {
            grid = grid1;
            int num_islands = 0;
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[0].length; j++) {
                    if (grid[i][j] == '1') {
                        num_islands++;
                        dfs_islands(i * grid[0].length + j);
                    }
                }
            }

            return num_islands;
        }

        public void dfs_islands(int start){
            int row = start / grid[0].length;
            int col = start % grid[0].length;
            grid[row][col] = '0';
            if (row + 1 < grid.length && grid[row + 1][col] == '1') {
                dfs_islands((row + 1) * grid[0].length + col);
            }
            if (row - 1 >= 0 && grid[row - 1][col] == '1') {
                dfs_islands((row - 1) * grid[0].length + col);
            }
            if (col + 1 > grid[0].length && grid[row][col + 1] == '1') {
                dfs_islands( (row) * grid[0].length + col + 1);
            }
            if (col - 1 >= 0 && grid[row][col - 1] == '1') {
                dfs_islands(row * grid[0].length + col - 1);

            }

        }
    }

    public static class Flood {
        int[][] image1;
        public int[][] floodfill(int[][] image, int sr, int sc, int newColor){
            image1 = image;
            dfs_flood(sr,sc,newColor);
            return image1;
        }

        public void dfs_flood(int r, int c, int newCol){
            int prev_col = image1[r][c];
            if(prev_col == newCol) return;
            image1[r][c] = newCol;
            if (r + 1 < image1.length && image1[r + 1][c] == prev_col) {
                dfs_flood(r + 1, c, newCol);
            }
            if (r - 1 >= 0 && image1[r - 1][c] == prev_col) {
                dfs_flood(r - 1, c, newCol);
            }
            if (c + 1 < image1[0].length && image1[r][c + 1] == prev_col) {
                dfs_flood(r, c + 1, newCol);
            }
            if (c - 1 >= 0 && image1[r][c - 1] == prev_col) {
                dfs_flood(r, c - 1, newCol);
            }

        }
    }


    public static class MergeLists {
        public ListNode mergeTwoLists(ListNode l1, ListNode l2){
            ListNode head = new ListNode(0), curr = head;

            while(l1 != null && l2 != null){
                if(l1.val < l2.val) {
                    curr.next = l1;
                    l1 = l1.next;
                } else {
                    curr.next = l2;
                    l2 = l2.next;
                }
                curr = curr.next;
            }

            curr.next = (l1 == null) ? l2 : l1;
            return head.next;
        }
    }

    public static class DefangIP {
        public String defangIPaddr(String address){
            int len = address.length();
            StringBuilder str = new StringBuilder();
            int i = 0;
            while(i < len){
                if(address.charAt(i) == '.'){
                    str.append("[.]");
                } else {
                    str.append(address.charAt(i));
                }
                i++;
            }
            return str.toString();
        }
    }

    public static class Node {
        public int val;
        public List<Node> children;
        public Node() {}
        public Node(int _val){
            val = _val;
        }
        public Node(int _val, List<Node> _children){
            val = _val;
            children = _children;
        }
    }

    public static class Depth {
        public int maxDepth(Node root){
            if(root == null){
                return 0;
            } else {
                int max = 0;
                for(int i = 0; i < root.children.size(); i++){
                    max = Math.max(max, maxDepth(root.children.get(i)));
                }
                return 1 + max;
            }
        }
    }

    public static class Balanced {
        // depth of right and left subtree of every node differ by at most one
        //
        public boolean isBalanced(TreeNode root){
            if(root == null){
                return true;
            } else {
                if(Math.abs(depth_dfs(root.left) - depth_dfs(root.right)) > 1) return false;
                return isBalanced(root.left) && isBalanced(root.right);
            }

        }

        private int depth_dfs(TreeNode a){
            if(a == null){
                return 0;
            } else {
                return 1 + Math.max(depth_dfs(a.left), depth_dfs(a.right));
            }
        }
    }

    public static class JewelsStones {
        public int numJewelsInStones(String J, String S) {
            HashSet<Character> jew = new HashSet<>();
            for(int i = 0; i < J.length(); i++){
                jew.add(J.charAt(i));
            }
            int res = 0;
            for(int i = 0; i < S.length(); i++){
                if(jew.contains(S.charAt(i))) res++;
            }
            return res;
        }
    }

    public static class sNumber {
        public int singleNumber(int[] nums) {
           int res = 0;
           for(int i : nums){
               res ^= i;
           }
           return res;
        }
    }

    public static class Codec {
        HashMap<Integer, String> hm = new HashMap<>();
        // Encodes a URL to a shortened URL.
        public String encode(String longUrl) {
            int sum = 0;
            for(Character c : longUrl.toCharArray()){
                sum += (int) c;
            }
            int url = 32978 % sum;
            hm.put(url, longUrl);

            return Integer.toString(url);

        }

        // Decodes a shortened URL to its original URL.
        public String decode(String shortUrl) {
            int code = Integer.valueOf(shortUrl);
            return hm.get(code);
        }
    }

    public static class Happy {
        public boolean isHappy(int n){
            HashSet<Integer> set = new HashSet<>();
            while(true){
                int sum = 0;
                while(n != 0){
                    int square = (int) Math.pow(n % 10, 2);
                    sum += square;
                    n = n/10;
                }
                if(sum == 1){
                    return true;
                } else if (set.contains(sum)) {
                    return false;
                } else {
                    set.add(sum);
                }
                n = sum;
            }
        }
    }

    public static class Fibonnaci {
        public int fibonnaci(int n){
            if(n == 0 || n == 1) return 1;
            return fibonnaci(n - 1) + fibonnaci(n - 2);
        }

        public int fibonacci_rec(int n){
            int[] fib = new int[n + 1];
            fib[0] = 1; fib[1] = 1;
            for(int i = 2; i < n + 1; i++){
                fib[i] = fib[i - 1] + fib[i - 2];
            }
            return fib[n];
        }

        public int fibonacci_short(int n){
            if(n == 0 || n == 1) return 1;
            int back = 1;
            int back1 = 1;
            int next = 0;
            int i = 2;
            while(i <= n){
                next = back + back1;
                back = back1;
                back1 = next;
            }
            return next;
        }
    }

    public static int longestSequence(int[] seq){
        int len = seq.length;
        int[] l = new int[len];
        l[0] = 1;
        for(int i = 1; i < len; i++){
            int max = 1;
            for(int j = 0; j < i; j++){
                if(seq[i] > seq[j]){
                    max = Math.max(max, l[j] + 1);
                }
            }
            l[i] = max;
        }
        int res = 0;
        for(int i : l) res = Math.max(res, i);
        return res;
    }

    public static class Stairs {
        public int climbStairs(int n){
            int back = 0;
            int back1 = 1;

            for(int i = 1; i < n; i++){
                int next = back + back1;
                back = back1;
                back1 = next;
            }
            return back + back1;
        }

    }

    public static class Robber {
        public int rob(int[] nums){
            int[] opt = new int[nums.length];
            opt[0] = nums[0];
            opt[1] = Math.max(nums[0], nums[1]);
            for(int i = 2; i < nums.length; i++){
                opt[i] = Math.max(opt[i - 1], nums[i] + opt[i - 2]);
            }
            return opt[nums.length - 1];
        }
    }

    public static int majorityElement(int[] nums){
        Integer cand = null;
        int count = 0;
        for(int i : nums){
            if(count == 0) cand = i;
            count += (i == cand)? 1 : -1;
        }
        return cand;
    }

    public static class Path {
        public boolean hasPathSum(TreeNode root, int sum){
            if(root == null) return false;
            if(sum - root.val == 0) return true;
            return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
        }
    }

    // Todo: work on this
    public static class ArrayToBST{
        public TreeNode sortedArrayToBST(int[] nums){
            return sortArray(nums, 0, nums.length - 1);
        }

        public TreeNode sortArray(int[] nums, int start, int end){
            if(start < end) {
                int mid = (start + end) / 2;
                TreeNode curr = new TreeNode(nums[mid]);
                curr.left = sortArray(nums, 0, start - 1);
                curr.right = sortArray(nums, start + 1, end);
                return curr;
            } else {
                return null;
            }
        }
    }

    public static class MaxArea {
        int[][] _grid;
        public int maxAreaOfIsland(int[][] grid) {
            int max = 0;
            _grid = grid;
            for(int i = 0; i < _grid.length; i++){
                for(int j = 0; j < _grid[0].length; j++){
                    if(grid[i][j] == 1){
                        int curr_max = dfs(i, j);
                        max = Math.max(curr_max, max);
                    }
                }
            }
            return max;
        }

        public int dfs(int row, int col){
            _grid[row][col] = 0;
            int count = 1;
            if(row + 1 < _grid.length && _grid[row + 1][col] == 1){
                count += dfs(row + 1, col);
            }
            if(row - 1 >= 0 && _grid[row - 1][col] == 1){
                count += dfs(row - 1, col);
            }
            if(col + 1 < _grid[0].length && _grid[row][col + 1] == 1){
                count += dfs(row, col + 1);
            }
            if(col - 1 >= 0 && _grid[row][col - 1] == 1){
                count += dfs(row, col - 1);
            }
            return count;
        }
    }

    public static void reverseString(char[] s) {
        int len = s.length;

        for(int i = 0; i < len / 2; i++) {
            char temp = s[i];
            s[i] = s[len - 1 - i];
            s[len - 1 - i] = temp;
        }

    }

    public static class Roman {
        public int romanToInt(String s){
            int len = s.length();
            int idx = 0;
            int sum = 0;
            while(idx < len){
                if(idx + 1 < len && get(s.charAt(idx)) < get(s.charAt(idx + 1))){
                    sum += get(s.charAt(idx + 1)) - get(s.charAt(idx));
                    idx += 2;
                } else {
                    sum += get(s.charAt(idx));
                    idx++;
                }
            }
            return sum;
        }

        public int get(char s){
            if(s == 'I') return 1;
            if(s == 'V') return 5;
            if(s == 'X') return 10;
            if(s == 'L') return 50;
            if(s == 'C') return 100;
            if(s == 'D') return 500;
            if(s == 'M') return 1000;
            else return -1;
        }
    }

    public static class Reverse {
        public ListNode reverseList(ListNode head) {
            ListNode prev = null;
            ListNode curr = head;
            ListNode next = null;
            while(curr != null){
                next = curr.next;
                curr.next = prev;
                prev = curr;
                curr = next;
            }
            return prev;
        }

        public ListNode reverse(ListNode head){
            if(head == null) return null;
            return helper(null, head);
        }

        public ListNode helper(ListNode head, ListNode next){
            if(next == null) return head;
            ListNode temp = next.next;
            next.next = head;
            return helper(next, temp);
        }


    }

    public static class Delete {
        public void deleteNode(ListNode node) {
            ListNode next = node.next;
            node.val = next.val;
            node.next = next.next;
        }

    }

    public static class SubSequence {
        public boolean isSubsequence(String s, String t){
            if(s.length() == 0) return true;
            if(t.length() == 0) return false;
            for(int i = 0; i < t.length(); i++){
                if(t.charAt(i) == s.charAt(0)){
                    return isSubsequence(s.substring(1), t.substring(i+1));
                }
            }
            return false;
        }
    }

    public static class Baseball {
        public int calPoints(String[] ops){
            Stack<String> st = new Stack<>();
            int sum = 0;
            for(int i = 0; i < ops.length; i++){
                if(ops[i].equals("C")){
                    int t = Integer.parseInt(st.pop());
                    sum -= t;
                } else if(ops[i].equals("D")){
                    int t = Integer.parseInt(st.peek());
                    t *= 2;
                    sum += t;
                    st.push(Integer.toString(t));
                } else if(ops[i].equals("+")){
                    int t1 = Integer.parseInt(st.pop());
                    int t2 = Integer.parseInt(st.pop());
                    int res = t1 + t2;
                    sum += res;
                    st.push(Integer.toString(t2)); st.push(Integer.toString(t1));
                    st.push(Integer.toString(res));
                } else {
                    st.push(ops[i]);
                    sum += Integer.parseInt(ops[i]);
                }
            }

            return sum;
        }
    }

    public static class TreeProb {

        public String solution(long[] arr) {
            // Type your solution here
            if(arr.length == 0) return "";
            long left = dfs(arr, 1);
            long right = dfs(arr, 2);
            if(left > right){
                return "Left";
            } else if (right > left){
                return "Right";
            } else {
                return "";
            }

        }

        public long dfs(long[] arr, int i){
            int len = arr.length;
            if(i > arr.length - 1 || arr[i] == -1) return 0;
            return arr[i] + dfs(arr, 2*i + 1) + dfs(arr, 2*i + 2);
        }
    }

    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = BigDecimal.valueOf(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }


    public static class Solution {
        public double solution(String[] aircraftEmissions, String[] flightItineraries, String origin, String destination) {
            // Type your solution here
            HashSet<String> hm = new HashSet<>();
            double dist = 0;
            for(int i = 0; i < flightItineraries.length; i++){
                String[] parts = flightItineraries[i].split("-");
                String orig = parts[0];
                String dest = parts[1];
                if(orig.equals(origin) && destination.equals(dest)){
                    dist = Double.parseDouble(parts[2]);
                    String plane = parts[3];
                    hm.add(plane);
                }
            }
            double emissions = Integer.MAX_VALUE;
            for(int i = 0; i < aircraftEmissions.length; i++){
                String[] parts = aircraftEmissions[i].split("-");
                String aircraftType = parts[0];
                if(hm.contains(aircraftType)){
                    emissions = Math.min(emissions, Double.parseDouble(parts[1]));
                }
            }
            double res = round(emissions * dist, 2);
            return res;
        }

        public double round(double value, int places) {
            BigDecimal bd = BigDecimal.valueOf(value);
            bd = bd.setScale(places, RoundingMode.HALF_UP);
            return bd.doubleValue();
        }
    }

    public static String[] solution(String[] availableProgrammes, long carbonToOffset) {
        // Type your solution here
        int k = 2;
        if(availableProgrammes.length <= 2) k = 1;
        TreeMap<Double, String> tm = new TreeMap<>();
        for(int i = 0; i < availableProgrammes.length; i++){
            double value = program(availableProgrammes[i], carbonToOffset);
            tm.put(value, availableProgrammes[i]);
        }
        String[] res = new String[k];
        int j = 0;
        for(Map.Entry<Double, String> entrySet : tm.entrySet()){
            if(j < k){
                res[j] = entrySet.getValue();
                j++;
            } else {
                break;
            }
        }
        return res;

    }

    public static double program(String program, long carbon){
        double res = 0;
        if(program.equals("s1")){
            if(carbon < 500){
                res = carbon * 1.25;
            } else {
                res = carbon;
            }
        }
        if(program.equals("s2")){
            if(carbon < 300){
                res = carbon * 1.5;
            } else {
                res = (300 * 1.5) + ((carbon - 300) * 0.5);
            }
        }
        if(program.equals("b1")){
            res = 0.8 * carbon;
        }
        if(program.equals("h1")){
            res = 100 + 0.5 * carbon;
        }
        if(program.equals("h2")){
            if(carbon % 10 == 0){
                res = (carbon / 10) * 9.90;
            } else {
                res = ((carbon + 1) / 10) * 9.90;
            }
        }
        return res;

    }

    public static ListNode reverseList(ListNode head){
        ListNode prev = null;
        ListNode curr = head;
        ListNode next = null;
        while(curr != null){
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    public static void main(String[] args) {

        CoinChange obj = new CoinChange();
        int[] coins = {2};
        int amount = 3;
        System.out.println(obj.coinChange(coins, amount));
    }
}
