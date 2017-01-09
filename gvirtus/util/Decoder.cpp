
/**
 * @file   Decoder.cpp
 * @author Raffaele Montella <raffaele.montella@uniparthenope.it>
 * @date   Wed Nov 30 17:16:48 2016
 *
 * @brief
 *
 *
 */

#include "Decoder.h"

#include <cstdio>
#include <iostream>


using namespace std;
using namespace log4cplus;


Decoder::Decoder() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("Decoder"));
    this->_buffersize=1024;
    this->step = step_a;
    this->plainchar = 0;
}

Decoder::Decoder(const Decoder& orig) {
}

Decoder::~Decoder() {
}

int Decoder::Decode(char value_in) {
    return Value(value_in);
}

int Decoder::Decode(const char* code_in, const int length_in, char* plaintext_out) {
    LOG4CPLUS_DEBUG(logger, "Decoder::Decode(const char*, const int, char *)" );
    return Block(code_in, length_in, plaintext_out);
}

void Decoder::Decode(std::istream& istream_in, std::ostream& ostream_in) {
    const int N = _buffersize;
    char* code = new char[N];
    char* plaintext = new char[N];
    int codelength;
    int plainlength;

    do
    {
        istream_in.read((char*)code, N);
        codelength = istream_in.gcount();
        plainlength = Decode(code, codelength, plaintext);
        ostream_in.write((const char*)plaintext, plainlength);
    }
    while (istream_in.good() && codelength > 0);

    this->step = step_a;
    this->plainchar = 0;

    delete [] code;
    delete [] plaintext;
}

int Decoder::Value(char value_in)
{
        static const char decoding[] = {62,-1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,-1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51};
        static const char decoding_size = sizeof(decoding);
        value_in -= 43;
        if (value_in < 0 || value_in > decoding_size) return -1;
        return decoding[(int)value_in];
}

int Decoder::Block(const char* code_in, const int length_in, char* plaintext_out)
{
    LOG4CPLUS_DEBUG(logger, "Decoder::Block" );
     const char* codechar = code_in;
        char* plainchar = plaintext_out;
        char fragment;

        *plainchar = this->plainchar;

        switch (this->step)
        {
                while (1)
                {
        case step_a:
                        do {
                                if (codechar == code_in+length_in)
                                {
                                        this->step = step_a;
                                        this->plainchar = *plainchar;
                                        return plainchar - plaintext_out;
                                }
                                fragment = (char)Value(*codechar++);
                        } while (fragment < 0);
                        *plainchar    = (fragment & 0x03f) << 2;
        case step_b:
                        do {
                                if (codechar == code_in+length_in)
                                {
                                        this->step = step_b;
                                        this->plainchar = *plainchar;
                                        return plainchar - plaintext_out;
                                }
                                fragment = (char)Value(*codechar++);
                        } while (fragment < 0);
                        *plainchar++ |= (fragment & 0x030) >> 4;
                        *plainchar    = (fragment & 0x00f) << 4;
        case step_c:
                        do {
                                if (codechar == code_in+length_in)
                                {
                                        this->step = step_c;
                                        this->plainchar = *plainchar;
                                        return plainchar - plaintext_out;
                                }
                                fragment = (char)Value(*codechar++);
                        } while (fragment < 0);
                        *plainchar++ |= (fragment & 0x03c) >> 2;
                        *plainchar    = (fragment & 0x003) << 6;
        case step_d:
                        do {
                                if (codechar == code_in+length_in)
                                {
                                        this->step = step_d;
                                        this->plainchar = *plainchar;
                                        return plainchar - plaintext_out;
                                }
                                fragment = (char)Value(*codechar++);
                        } while (fragment < 0);
                        *plainchar++   |= (fragment & 0x03f);
                }
        }
        /* control should not reach here */
        return plainchar - plaintext_out;
 
}



