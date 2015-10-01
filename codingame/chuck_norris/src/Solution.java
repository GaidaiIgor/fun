import java.io.FileInputStream;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

class Solution
{
    public static String get_bits(String message)
    {
        StringBuilder result = new StringBuilder();
        for (char next: message.toCharArray())
        {
            String bits = String.format("%7s", Integer.toBinaryString(next)).replace(' ', '0');
            result.append(bits);
        }
        return result.toString();
    }

    public static List<String> split(String bits)
    {
        char current = bits.charAt(0);
        StringBuilder token_builder = new StringBuilder();
        List<String> result = new ArrayList<>();
        for (char next_bit: bits.toCharArray())
        {
            if (next_bit != current)
            {
                result.add(token_builder.toString());
                token_builder.setLength(0);
                current = next_bit;
            }
            token_builder.append(next_bit);
        }
        result.add(token_builder.toString());
        return result;
    }

    public static String encode(String bits)
    {
        StringBuilder encoded = new StringBuilder();
        List<String> splitted = split(bits);
        for (String token: splitted)
        {
            String first = token.charAt(0) == '0' ? "00 " : "0 ";
            String second = Collections.nCopies(token.length(), "0").stream().collect(Collectors.joining()) + " ";
            encoded.append(first);
            encoded.append(second);
        }
        return encoded.toString();
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
        String message = in.nextLine();
        String binary = get_bits(message);
        String encoded = encode(binary);
        System.out.println(encoded.trim());
    }
}