public class CoinChange {
    public int coinChange(int[] coins, int amount) {
        int[] opt = new int[amount + 1];
        opt[0] = 0;
        for (int i = 1; i < opt.length; i++) {
            opt[i] = Integer.MAX_VALUE;
            for (int j = 0; j < coins.length; j++) {
                if (i - coins[j] >= 0 && opt[i - coins[j]] != -1) {
                    opt[i] = Math.min(opt[i], 1 + opt[i - coins[j]]);
                }
            }
            if (opt[i] == Integer.MAX_VALUE) opt[i] = -1;
        }
        return opt[amount];

    }
}
