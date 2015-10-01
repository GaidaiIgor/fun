import java.io.FileInputStream;
import java.util.*;

class Solution
{
    public static Map<Character, List<String>> get_ascii_map(int width, int height, Scanner in)
    {
        Map<Character, List<String>> result = new HashMap<>();
        String keys = "ABCDEFGHIJKLMNOPQRSTUVWXYZ?";
        for (int i = 0; i < height; ++i)
        {
            String next_line = in.nextLine();
            for (int j = 0; j < keys.length(); ++j)
            {
                char current = keys.charAt(j);
                if (!result.containsKey(current))
                {
                    result.put(current, new ArrayList<>());
                }
                result.get(current).add(next_line.substring(j * width, (j + 1) * width));
            }
        }
        return result;
    }

    public static void print_text(Map<Character, List<String>> ascii_map, String text)
    {
        text = text.toUpperCase();
        for (int i = 0; i < ascii_map.get('A').size(); ++i)
        {
            for (char next: text.toCharArray())
            {
                if (ascii_map.containsKey(next))
                {
                    System.out.print(ascii_map.get(next).get(i));
                }
                else
                {
                    System.out.print(ascii_map.get('?').get(i));
                }
            }
            System.out.print("\n");
        }
    }

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
        int width = in.nextInt();
        int height = in.nextInt();
        in.nextLine();
        String text = in.nextLine();
        Map<Character, List<String>> ascii_map = get_ascii_map(width, height, in);
        print_text(ascii_map, text);
    }
}