/**
 * @file   Decoder.h
 * @author Raffaele Montella <raffaele.montella@uniparthenope.it>
 * @date   Wed Nov 30 17:16:48 2016
 * 
 * @brief  
 * 
 * 
 */

#ifndef _ENCODER_H
#define	_ENCODER_H

#include <iostream>
#include <cstdlib>

typedef enum
{
        step_A, step_B, step_C
} EncodeStep;

class Encoder {
public:
    Encoder();
    Encoder(const Encoder& orig);
    virtual ~Encoder();

    int Encode(char value_in);
    int Encode(const char*, int, char*);
    void Encode(std::istream&, std::ostream&);

private:
    int CHARS_PER_LINE;

    int _buffersize;
    EncodeStep step;
    char result;
    int stepcount;

    int Value(char value_in);
    int Block(const char* code_in, const int length_in, char* plaintext_out);
    int BlockEnd(char* code_out);
    int EncodeEnd(char* code_out);
};

#endif	/* _ENCODER_H */

