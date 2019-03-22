
/**
 * @file   Encoder.cpp
 * @author Raffaele Montella <raffaele.montella@uniparthenope.it>
 * @date   Wed Nov 30 17:16:48 2016
 *
 * @brief
 *
 *
 */

#include "Encoder.h"

#include <cstdio>
#include <iostream>


using namespace std;

Encoder::Encoder() {
    CHARS_PER_LINE = 72;
    step = step_A;
    result = 0;
    stepcount = 0;
}

Encoder::Encoder(const Encoder& orig) {
}

Encoder::~Encoder() {
}

int Encoder::Encode(char value_in) {
    return Value(value_in);
}

int Encoder::Encode(const char* code_in, const int length_in, char* plaintext_out) {
    return Block(code_in, length_in, plaintext_out);
}

int Encoder::EncodeEnd(char* plaintext_out)
{
    return BlockEnd(plaintext_out);
}




int Encoder::Value(char value_in)
{
    static const char* encoding = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (value_in > 63) return '=';
    return encoding[(int)value_in];
}

int Encoder::Block(const char* plaintext_in, int length_in, char* code_out)
{
      const char* plainchar = plaintext_in;
        const char* const plaintextend = plaintext_in + length_in;
        char* codechar = code_out;
        char result;
        char fragment;

        result = this->result;

        switch (this->step)
        {
                while (1)
                {
        case step_A:
                        if (plainchar == plaintextend)
                        {
                                this->result = result;
                                this->step = step_A;
                                return codechar - code_out;
                        }
                        fragment = *plainchar++;
                        result = (fragment & 0x0fc) >> 2;
                        *codechar++ = Value(result);
                        result = (fragment & 0x003) << 4;
        case step_B:
                        if (plainchar == plaintextend)
                        {
                                this->result = result;
                                this->step = step_B;
                                return codechar - code_out;
                        }
                        fragment = *plainchar++;
                        result |= (fragment & 0x0f0) >> 4;
                        *codechar++ = Value(result);
                        result = (fragment & 0x00f) << 2;
        case step_C:
                        if (plainchar == plaintextend)
                        {
                                this->result = result;
                                this->step = step_C;
                                return codechar - code_out;
                        }
                        fragment = *plainchar++;
                        result |= (fragment & 0x0c0) >> 6;
                        *codechar++ = Value(result);
                        result  = (fragment & 0x03f) >> 0;
                        *codechar++ = Value(result);

                        ++(this->stepcount);
                        if (this->stepcount == CHARS_PER_LINE/4)
                        {
                                *codechar++ = '\n';
                                this->stepcount = 0;
                        }
                }
        }
        /* control should not reach here */
        return codechar - code_out;

}

int Encoder::BlockEnd(char* code_out)
{
        char* codechar = code_out;

        switch (this->step)
        {
        case step_B:
                *codechar++ = Value(this->result);
                *codechar++ = '=';
                *codechar++ = '=';
                break;
        case step_C:
                *codechar++ = Value(this->result);
                *codechar++ = '=';
                break;
        case step_A:
                break;
        }
        *codechar++ = '\n';

        return codechar - code_out;
}

void Encoder::Encode(std::istream& istream_in, std::ostream& ostream_in)
{
    step = step_A;
    result = 0;
    stepcount = 0;           
                        //
                        const int N = _buffersize;
                        char* plaintext = new char[N];
                        char* code = new char[2*N];
                        int plainlength;
                        int codelength;

                        do
                        {
                                istream_in.read(plaintext, N);
                                plainlength = istream_in.gcount();
                                //
                                codelength = Encode(plaintext, plainlength, code);
                                ostream_in.write(code, codelength);
                        }
                        while (istream_in.good() && plainlength > 0);

                        codelength = EncodeEnd(code);
                        ostream_in.write(code, codelength);
                        //
    step = step_A;
    result = 0;
    stepcount = 0;

    delete [] code;
    delete [] plaintext;
}

