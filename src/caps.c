// Enumerate Vulkan cooperative-matrix capabilities on the current GPU.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <vulkan/vulkan.h>

static const char* ctype(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
        case VK_COMPONENT_TYPE_SINT8_KHR:   return "s8";
        case VK_COMPONENT_TYPE_UINT8_KHR:   return "u8";
        case VK_COMPONENT_TYPE_SINT16_KHR:  return "s16";
        case VK_COMPONENT_TYPE_UINT16_KHR:  return "u16";
        case VK_COMPONENT_TYPE_SINT32_KHR:  return "s32";
        case VK_COMPONENT_TYPE_UINT32_KHR:  return "u32";
#ifdef VK_COMPONENT_TYPE_BFLOAT16_KHR
        case VK_COMPONENT_TYPE_BFLOAT16_KHR: return "bf16";
#endif
        default: return "?";
    }
}
static const char* scope(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR: return "DEVICE";
        case VK_SCOPE_WORKGROUP_KHR: return "WORKGROUP";
        case VK_SCOPE_SUBGROUP_KHR: return "SUBGROUP";
        case VK_SCOPE_QUEUE_FAMILY_KHR: return "QUEUEFAM";
        default: return "?";
    }
}
#define CHECK(x) do { VkResult r_=(x); if(r_!=VK_SUCCESS){fprintf(stderr,"%s -> %d\n",#x,(int)r_); return 1;} } while(0)

int main(void) {
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &ai;
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, NULL, &inst));

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pGetCMP =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!pGetCMP) { fprintf(stderr, "no vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\n"); }

    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
    VkPhysicalDevice* devs = malloc(sizeof(*devs) * (n ? n : 1));
    CHECK(vkEnumeratePhysicalDevices(inst, &n, devs));
    printf("devices: %u\n", n);

    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        printf("\n== device %u: %s (api %u.%u.%u, driver 0x%08x) ==\n", i, p.deviceName,
               VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion),
               VK_VERSION_PATCH(p.apiVersion), p.driverVersion);

        uint32_t en = 0;
        vkEnumerateDeviceExtensionProperties(devs[i], NULL, &en, NULL);
        VkExtensionProperties* ex = malloc(sizeof(*ex) * (en ? en : 1));
        vkEnumerateDeviceExtensionProperties(devs[i], NULL, &en, ex);
        int hasCmat = 0, hasDot = 0, hasF16int8 = 0;
        for (uint32_t e = 0; e < en; e++) {
            if (!strcmp(ex[e].extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) hasCmat = 1;
            if (!strcmp(ex[e].extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) hasDot = 1;
            if (!strcmp(ex[e].extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) hasF16int8 = 1;
        }
        free(ex);
        printf("ext: cooperative_matrix=%d integer_dot_product=%d float16_int8=%d\n",
               hasCmat, hasDot, hasF16int8);

        VkPhysicalDeviceShaderFloat16Int8Features f16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES };
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
        dot.pNext = &cm;
        f16.pNext = &dot;
        VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        f2.pNext = &f16;
        vkGetPhysicalDeviceFeatures2(devs[i], &f2);
        printf("features: shaderFloat16=%d shaderInt8=%d integerDotProduct=%d cooperativeMatrix=%d\n",
               f16.shaderFloat16, f16.shaderInt8, dot.shaderIntegerDotProduct, cm.cooperativeMatrix);

        VkPhysicalDeviceSubgroupProperties sg = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(devs[i], &p2);
        printf("subgroup: size=%u stages=0x%x ops=0x%x\n",
               sg.subgroupSize, sg.supportedStages, sg.supportedOperations);

        if (hasCmat) {
            uint32_t cn = 0;
            VkResult r = pGetCMP(devs[i], &cn, NULL);
            printf("cooperative matrix configs: count=%u (query r=%d)\n", cn, (int)r);
            if (cn) {
                VkCooperativeMatrixPropertiesKHR* cp = calloc(cn, sizeof(*cp));
                for (uint32_t c = 0; c < cn; c++)
                    cp[c].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                CHECK(pGetCMP(devs[i], &cn, cp));
                for (uint32_t c = 0; c < cn; c++)
                    printf("  [%2u] M=%2u N=%2u K=%2u A=%-4s B=%-4s C=%-4s R=%-4s sat=%u scope=%s\n",
                           c, cp[c].MSize, cp[c].NSize, cp[c].KSize,
                           ctype(cp[c].AType), ctype(cp[c].BType),
                           ctype(cp[c].CType), ctype(cp[c].ResultType),
                           cp[c].saturatingAccumulation, scope(cp[c].scope));
                free(cp);
            }
        }
    }
    vkDestroyInstance(inst, NULL);
    return 0;
}
