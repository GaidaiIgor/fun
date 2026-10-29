// Solves CodinGame Nintendo Sponsored Contest. The encoder multiplies the polynomials over GF(2) held in the two halves of the input bits,
// so the decoder factors the message polynomial into irreducible polynomials and prints every split of the factors into two products
// fitting the halves.
#include <bits/stdc++.h>
using namespace std;

// Holds a polynomial over GF(2) with the coefficient of x^i in bit i, wide enough for the square of a remainder modulo a message.
using Poly = bitset<1024>;

vector<Poly> factorize(Poly poly);
Poly gcd(Poly a, Poly b);
Poly mul(const Poly& a, const Poly& b);
pair<Poly, Poly> divide(Poly a, const Poly& b);
int deg(const Poly& poly);

// Reads the size and the message words, holding the coefficient of x^i in bit i % 32 of word i / 32, and multiplies the irreducible
// factors of the message into the set divisors of all its divisors. Every divisor low whose quotient high leaves both of them below
// size bits gives the input words, low in the first half and high in the second, and the inputs are printed in alphabetical order.
int main() {
    int size;
    cin >> size;
    Poly message;
    for (int k = 0; k < size / 16; k++) {
        unsigned word;
        cin >> hex >> word;
        message |= Poly(word) << 32 * k;
    }
    unordered_set<Poly> divisors{Poly(1)};
    for (const Poly& factor : factorize(message))
        for (const Poly& divisor : vector<Poly>(divisors.begin(), divisors.end()))
            divisors.insert(mul(divisor, factor));
    vector<string> inputs;
    for (const Poly& low : divisors) {
        Poly high = divide(message, low).first;
        if (max(deg(low), deg(high)) >= size)
            continue;
        string line;
        for (int k = 0; k < size / 16; k++) {
            char text[10];
            snprintf(text, sizeof text, k ? " %08lx" : "%08lx", ((low | high << size) >> 32 * k & Poly(0xFFFFFFFF)).to_ulong());
            line += text;
        }
        inputs.push_back(line);
    }
    sort(inputs.begin(), inputs.end());
    for (const string& line : inputs)
        cout << line << endl;
}

// Factors the polynomial poly into irreducible polynomials. For every degree, the gcd of poly and x^(2^degree) - x, where power is
// x^(2^degree) modulo poly, is the product of the distinct irreducible factors of that degree, since poly has lost all its factors of
// lower degrees by then. The products in pending are split by the Cantor-Zassenhaus method, as the gcd of a product group and the trace
// of a random polynomial term modulo it, the sum of its 2^i powers below 2^degree, holds the factors where the trace is 0, about half
// of them. Every irreducible factor is divided out of poly as many times as it divides it. Once poly is shorter than two factors of the
// next degree, it is irreducible or 1. Returns the list of the irreducible factors repeated by their multiplicities.
vector<Poly> factorize(Poly poly) {
    vector<Poly> factors;
    Poly power(2);
    mt19937 random;
    for (int degree = 1; deg(poly) >= 2 * degree; degree++) {
        power = divide(mul(power, power), poly).second;
        vector<Poly> pending{gcd(power ^ Poly(2), poly)};
        while (!pending.empty()) {
            Poly group = pending.back();
            pending.pop_back();
            if (int length = deg(group); length > degree) {
                Poly term;
                for (int i = 0; i < length; i++)
                    term[i] = random() & 1;
                Poly trace = term;
                for (int i = 1; i < degree; i++)
                    trace ^= (term = divide(mul(term, term), group).second);
                if (Poly part = gcd(trace, group); deg(part) > 0 && deg(part) < length)
                    pending.insert(pending.end(), {part, divide(group, part).first});
                else
                    pending.push_back(group);
            } else if (length > 0)
                for (auto division = divide(poly, group); division.second.none(); division = divide(poly, group)) {
                    poly = division.first;
                    factors.push_back(group);
                }
        }
    }
    if (deg(poly) > 0)
        factors.push_back(poly);
    return factors;
}

// Computes the greatest common divisor of the polynomials a and b and returns it, which is a if b is 0.
Poly gcd(Poly a, Poly b) {
    while (b.any())
        a = exchange(b, divide(a, b).second);
    return a;
}

// Multiplies the polynomials a and b and returns the product.
Poly mul(const Poly& a, const Poly& b) {
    Poly product;
    for (size_t i = 0; i < b.size(); i++)
        if (b[i])
            product ^= a << i;
    return product;
}

// Divides the polynomial a by the nonzero polynomial b and returns the quotient and the remainder.
pair<Poly, Poly> divide(Poly a, const Poly& b) {
    Poly quotient;
    for (int top = deg(a), bottom = deg(b); top >= bottom; top--)
        if (a[top]) {
            quotient[top - bottom] = true;
            a ^= b << (top - bottom);
        }
    return {quotient, a};
}

// Finds the degree of the polynomial poly by binary search for the largest shift leaving a nonzero polynomial and returns it, -1 for 0.
int deg(const Poly& poly) {
    int degree = poly.any() - 1;
    for (int step = poly.size() / 2; step; step /= 2)
        if ((poly >> (degree + step)).any())
            degree += step;
    return degree;
}
