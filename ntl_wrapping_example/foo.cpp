#include <iostream>
#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>
#include <string.h>
#include <NTL/ZZ.h>

using namespace std;
using namespace NTL;

class Foo{
public:
    void handleErrors(void)
        {
            ERR_print_errors_fp(stderr);
            abort();
        }

    void bar(){
        std::cout << "hello word" << std::endl;
    }

    int encrypt(unsigned char *plaintext, int plaintext_len, unsigned char *aad,
                int aad_len, unsigned char *key, unsigned char *iv,
                unsigned char *ciphertext, unsigned char *tag)
        {
            EVP_CIPHER_CTX *ctx;

            int len;

            int ciphertext_len;


            /* Create and initialise the context */
            if(!(ctx = EVP_CIPHER_CTX_new())) handleErrors();

            /* Initialise the encryption operation. */
            if(1 != EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL))
                handleErrors();

            /* Set IV length if default 12 bytes (96 bits) is not appropriate */
            if(1 != EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 16, NULL))
                handleErrors();

            /* Initialise key and IV */
            if(1 != EVP_EncryptInit_ex(ctx, NULL, NULL, key, iv)) handleErrors();

            /* Provide any AAD data. This can be called zero or more times as
             * required
             */
            if(1 != EVP_EncryptUpdate(ctx, NULL, &len, aad, aad_len))
                handleErrors();

            /* Provide the message to be encrypted, and obtain the encrypted output.
             * EVP_EncryptUpdate can be called multiple times if necessary
             */
            if(1 != EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, plaintext_len))
                handleErrors();
            ciphertext_len = len;

            /* Finalise the encryption. Normally ciphertext bytes may be written at
             * this stage, but this does not occur in GCM mode
             */
            if(1 != EVP_EncryptFinal_ex(ctx, ciphertext + len, &len)) handleErrors();
            ciphertext_len += len;

            /* Get the tag */
            if(1 != EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag))
                handleErrors();

            /* Clean up */
            EVP_CIPHER_CTX_free(ctx);

            return ciphertext_len;
        }

    int decrypt(unsigned char *ciphertext, int ciphertext_len, unsigned char *aad,
                int aad_len, unsigned char *tag, unsigned char *key, unsigned char *iv,
                unsigned char *plaintext)
        {
            EVP_CIPHER_CTX *ctx;
            int len;
            int plaintext_len;
            int ret;

            /* Create and initialise the context */
            if(!(ctx = EVP_CIPHER_CTX_new())) handleErrors();

            /* Initialise the decryption operation. */
            if(!EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, NULL, NULL))
                handleErrors();

            /* Set IV length. Not necessary if this is 12 bytes (96 bits) */
            if(!EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 16, NULL))
                handleErrors();

            /* Initialise key and IV */
            if(!EVP_DecryptInit_ex(ctx, NULL, NULL, key, iv)) handleErrors();

            /* Provide any AAD data. This can be called zero or more times as
             * required
             */
            if(!EVP_DecryptUpdate(ctx, NULL, &len, aad, aad_len))
                handleErrors();

            /* Provide the message to be decrypted, and obtain the plaintext output.
             * EVP_DecryptUpdate can be called multiple times if necessary
             */
            if(!EVP_DecryptUpdate(ctx, plaintext, &len, ciphertext, ciphertext_len))
                handleErrors();
            plaintext_len = len;

            /* Set expected tag value. Works in OpenSSL 1.0.1d and later */
            if(!EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, tag))
                handleErrors();

            /* Finalise the decryption. A positive return value indicates success,
             * anything else is a failure - the plaintext is not trustworthy.
             */
            ret = EVP_DecryptFinal_ex(ctx, plaintext + len, &len);

            /* Clean up */
            EVP_CIPHER_CTX_free(ctx);

            if(ret > 0)
            {
                /* Success */
                plaintext_len += len;
                return plaintext_len;
            }
            else
            {
                /* Verify failed */
                return -1;
            }
        }

    int testencrypt (void)
        {
            /* Set up the key and iv. Do I need to say to not hard code these in a
             * real application? :-)
             */

            /* A 256 bit key */
            unsigned char *key = (unsigned char *)"01234567890123456789012345678901";

            /* A 128 bit IV */
            unsigned char *iv = (unsigned char *)"0123456789012345";

            /* Message to be encrypted */
            unsigned char *plaintext =
                (unsigned char *)"The quick brown fox jumps over the lazy dog";

            unsigned char *aad = (unsigned char *)"The associated data";
            int add_len = strlen((char *)plaintext);
            unsigned char tag[128];
            /* Buffer for ciphertext. Ensure the buffer is long enough for the
             * ciphertext which may be longer than the plaintext, dependant on the
             * algorithm and mode
             */
            unsigned char ciphertext[128];

            /* Buffer for the decrypted text */
            unsigned char decryptedtext[128];

            int decryptedtext_len, ciphertext_len;

            /* Initialise the library */
            ERR_load_crypto_strings();
            OpenSSL_add_all_algorithms();
            OPENSSL_config(NULL);

            /* Encrypt the plaintext */
            ciphertext_len = encrypt (plaintext, strlen ((char *)plaintext),
                                      aad, add_len, key, iv,
                                      ciphertext, tag);

            /* Do something useful with the ciphertext here */
            printf("Ciphertext is:\n");
            BIO_dump_fp (stdout, (const char *)ciphertext, ciphertext_len);

            /* Decrypt the ciphertext */
            decryptedtext_len = decrypt(ciphertext, ciphertext_len, aad,
                                        add_len, tag, key, iv, decryptedtext);

            /* Add a NULL terminator. We are expecting printable text */
            decryptedtext[decryptedtext_len] = '\0';

            /* Show the decrypted text */
            printf("Decrypted text is:\n");
            printf("%s\n", decryptedtext);

            /* Clean up */
            EVP_cleanup();
            ERR_free_strings();

            return 0;
        }

    void test_ntt(){
        ZZ a, b, c;
        cout << "Enter a: " << "\n";
        cin >> a;

        cout << "Enter b: " << "\n";
        cin >> b;

        c = (a+1)*(b+1);

        cout << c << "\n";
    }

};

extern "C"{
    Foo* Foo_new(){return new Foo();}
    void Foo_bar(Foo* foo){foo->bar();}
    void test_ntt(Foo* foo){foo->test_ntt();}
    void test_enc(Foo* foo){foo->testencrypt();}
    int enc(Foo* t, unsigned char
            *plaintext, int plaintext_len, unsigned
            char *aad, int aad_len, unsigned char *key,
            unsigned char *iv, unsigned char
            *ciphertext, unsigned char *tag){
        return t->encrypt(plaintext, plaintext_len, aad, aad_len, key,
                   iv, ciphertext, tag);
    }
    int dec(Foo* t, unsigned char
            *ciphertext, int ciphertext_len, unsigned
            char *aad, int aad_len, unsigned char
            *tag, unsigned char *key, unsigned char
            *iv, unsigned char *plaintext){
        return t->decrypt(ciphertext, ciphertext_len, aad, aad_len, tag,
                   key, iv, plaintext);
    }
}
