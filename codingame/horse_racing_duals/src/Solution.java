import java.io.FileInputStream;
import java.util.*;
import java.util.stream.IntStream;

class Solution
{
    public static void main(String args[])
    {
        try
        {
            System.setIn(new FileInputStream("input"));
        }
        catch (Exception ignored)
        {
        }

        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        List<Integer> strengths = new ArrayList<>();
        for (int i = 0; i < n; ++i)
        {
            int next_strength = in.nextInt();
            strengths.add(next_strength);
        }
        Collections.sort(strengths);
        System.out.println(IntStream.range(0, strengths.size() - 1).map(i -> Math.abs(strengths.get(i) - strengths.get(i + 1))).min().
                getAsInt());
    }
}