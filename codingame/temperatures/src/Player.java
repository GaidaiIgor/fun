import java.io.FileInputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

class Solution
{
    public static void main(String args[])
    {
        try
        {
            System.setIn(new FileInputStream("input"));
        }
        catch (Exception e)
        {
        }

        Scanner in = new Scanner(System.in);
        int n = in.nextInt(); // the number of temperatures to analyse
        if (n == 0)
        {
            System.out.println("0");
            return;
        }

        in.nextLine();
        String next_line = in.nextLine(); // the N temperatures expressed as integers ranging from -273 to 5526
        List<Integer> temperatures = Arrays.asList(next_line.split(" ")).stream().map(Integer::parseInt).collect(Collectors.toList());
        int result = temperatures.stream().
                min((a, b) -> Math.abs(a) == Math.abs(b) ? Integer.compare(-a, -b) : Integer.compare(Math.abs(a), Math.abs(b))).get();
        System.out.println(result);
    }
}