// Minimal Vulkan GEMM benchmark: XMX (cooperative matrix) vs shader cores,
// int8 (incl. an OpSDot DP4A baseline) and fp16. Timing uses GPU timestamps.
// No descriptor sets: buffers are passed to shaders as device addresses
// (buffer_reference) in push constants.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <vulkan/vulkan.h>

#define VC(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){ fprintf(stderr,"%s -> %d\n",#x,(int)_r); exit(1);} } while(0)

typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceAddress addr; VkDeviceSize size; } GpuBuf;

static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;
static VkPhysicalDeviceMemoryProperties memprops;
static float timestampPeriod = 1.0f;
static PFN_vkGetBufferDeviceAddress pGetBufAddr;

static uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
        if ((bits & (1u<<i)) && (memprops.memoryTypes[i].propertyFlags & props) == props) return i;
    fprintf(stderr, "no suitable memory type\n"); exit(1);
}

static GpuBuf make_buf(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mp) {
    GpuBuf b = {0}; b.size = size;
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VC(vkCreateBuffer(dev, &bi, NULL, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkMemoryAllocateFlagsInfo af = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    af.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.pNext = &af; ai.allocationSize = mr.size; ai.memoryTypeIndex = find_mem(mr.memoryTypeBits, mp);
    VC(vkAllocateMemory(dev, &ai, NULL, &b.mem));
    VC(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo di = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        di.buffer = b.buf; b.addr = pGetBufAddr(dev, &di);
    }
    return b;
}

static VkShaderModule load_spv(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t* code = malloc(n); if (fread(code, 1, n, f) != (size_t)n) { fprintf(stderr,"read fail\n"); exit(1);} fclose(f);
    VkShaderModuleCreateInfo ci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = n; ci.pCode = code;
    VkShaderModule m; VC(vkCreateShaderModule(dev, &ci, NULL, &m)); free(code); return m;
}

static void run_cmd(void (*rec)(VkCommandBuffer, void*), void* ud, VkFence fence) {
    VkCommandPoolCreateInfo cp = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO }; cp.queueFamilyIndex = qfam;
    VkCommandPool pool; VC(vkCreateCommandPool(dev, &cp, NULL, &pool));
    VkCommandBufferAllocateInfo ca = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ca.commandPool = pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VkCommandBuffer cb; VC(vkAllocateCommandBuffers(dev, &ca, &cb));
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VC(vkBeginCommandBuffer(cb, &bi));
    rec(cb, ud);
    VC(vkEndCommandBuffer(cb));
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VC(vkQueueSubmit(queue, 1, &si, fence));
    VC(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX)); vkResetFences(dev, 1, &fence);
    vkDestroyCommandPool(dev, pool, NULL);
}

typedef struct { GpuBuf* dst; const void* data; VkDeviceSize size; } UpArgs;
static void rec_upload(VkCommandBuffer cb, void* p) {
    UpArgs* a = p;
    VkBufferCopy r = {0,0,a->size}; vkCmdCopyBuffer(cb, *(VkBuffer*)a->data, a->dst->buf, 1, &r);
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
static void upload(GpuBuf* dst, const void* data, VkDeviceSize size) {
    GpuBuf st = make_buf(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* p; VC(vkMapMemory(dev, st.mem, 0, size, 0, &p)); memcpy(p, data, size); vkUnmapMemory(dev, st.mem);
    VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence f; VC(vkCreateFence(dev, &fi, NULL, &f));
    UpArgs ua = { dst, &st.buf, size };
    run_cmd(rec_upload, &ua, f);
    vkDestroyFence(dev, f, NULL);
    vkDestroyBuffer(dev, st.buf, NULL); vkFreeMemory(dev, st.mem, NULL);
}

typedef struct { GpuBuf* src; GpuBuf* st; VkDeviceSize size; } RbArgs;
static void rec_copy(VkCommandBuffer cb, void* p) {
    RbArgs* a = p;
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkBufferCopy r = {0,0,a->size}; vkCmdCopyBuffer(cb, a->src->buf, a->st->buf, 1, &r);
}
static void readback(GpuBuf* src, void* out, VkDeviceSize size) {
    GpuBuf st = make_buf(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence f; VC(vkCreateFence(dev, &fi, NULL, &f));
    RbArgs a = { src, &st, size };
    run_cmd(rec_copy, &a, f);
    vkDestroyFence(dev, f, NULL);
    void* p; VC(vkMapMemory(dev, st.mem, 0, size, 0, &p)); memcpy(out, p, size); vkUnmapMemory(dev, st.mem);
    vkDestroyBuffer(dev, st.buf, NULL); vkFreeMemory(dev, st.mem, NULL);
}

typedef struct { const char* name; const char* spv; int f16; int style; // style: 0=xmx,1=tile16
                 VkPipeline pipe; VkPipelineLayout pl; uint32_t gx, gy; } Kern;

static void build_kernel(Kern* k) {
    VkPushConstantRange pc = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 40 };
    VkPipelineLayoutCreateInfo pl = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pc;
    VC(vkCreatePipelineLayout(dev, &pl, NULL, &k->pl));
    VkShaderModule m = load_spv(k->spv);
    VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; ci.stage.module = m; ci.stage.pName = "main";
    ci.layout = k->pl;
    VC(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &k->pipe));
    vkDestroyShaderModule(dev, m, NULL);
}

static void pc_push(VkCommandBuffer cb, VkPipelineLayout pl, GpuBuf* A, GpuBuf* B, GpuBuf* C, uint32_t M, uint32_t N, uint32_t K) {
    struct { uint64_t a,b,c; uint32_t M,N,K; } pc = { A->addr, B->addr, C->addr, M, N, K };
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
}

// CPU half <-> float
static uint16_t f2h(float f){ uint32_t x; memcpy(&x,&f,4); uint32_t s=(x>>16)&0x8000; int32_t e=((x>>23)&0xff)-127+15; uint32_t m=(x>>13)&0x3ff;
    if(e<=0) return s; if(e>=31) return s|0x7c00; return s|(e<<10)|m; }
static float h2f(uint16_t h){ uint32_t s=(h&0x8000)<<16; int32_t e=(h>>10)&0x1f; uint32_t m=h&0x3ff; uint32_t x;
    if(e==0){ if(m==0){x=s;} else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; x=s|(e<<23)|(m<<13);} }
    else if(e==31){ x=s|0x7f800000|(m<<13);} else { x=s|((e-15+127)<<23)|(m<<13);} float f; memcpy(&f,&x,4); return f; }

typedef struct { Kern* k; GpuBuf* A; GpuBuf* B; GpuBuf* C; uint32_t M,N,K; int reps; VkQueryPool qp; } DispArgs;
static void rec_dispatch(VkCommandBuffer cb, void* p) {
    DispArgs* d = p;
    vkCmdResetQueryPool(cb, d->qp, 0, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, d->qp, 0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->k->pipe);
    pc_push(cb, d->k->pl, d->A, d->B, d->C, d->M, d->N, d->K);
    for (int r = 0; r < d->reps; r++) vkCmdDispatch(cb, d->k->gx, d->k->gy, 1);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, d->qp, 1);
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 2048;
    int N = argc > 2 ? atoi(argv[2]) : 2048;
    int K = argc > 3 ? atoi(argv[3]) : 2048;
    if (M % 32 || N % 32 || K % 32) { fprintf(stderr, "M,N,K must be multiples of 32\n"); return 1; }

    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.apiVersion = VK_API_VERSION_1_3; ai.pApplicationName = "xmx-bench";
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO }; ici.pApplicationInfo = &ai;
    VC(vkCreateInstance(&ici, NULL, &inst));
    uint32_t np = 1; VC(vkEnumeratePhysicalDevices(inst, &np, &phys));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    timestampPeriod = props.limits.timestampPeriod;
    printf("GPU: %s (api %u.%u.%u, timestampPeriod %.2f ns)\n", props.deviceName,
           VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion), timestampPeriod);

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties* qp = malloc(sizeof(*qp)*nq); vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp);
    qfam = UINT32_MAX; for (uint32_t i=0;i<nq;i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT){ qfam=i; break; }
    if (qfam==UINT32_MAX) { fprintf(stderr,"no compute queue\n"); return 1; }
    vkGetPhysicalDeviceMemoryProperties(phys, &memprops);

    VkPhysicalDeviceVulkan11Features v11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    v11.storageBuffer16BitAccess = VK_TRUE; v11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceVulkan13Features v13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    v13.shaderIntegerDotProduct = VK_TRUE; v13.pNext = &v11;
    VkPhysicalDeviceVulkan12Features v12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    v12.pNext = &v13; v12.shaderInt8 = VK_TRUE; v12.storageBuffer8BitAccess = VK_TRUE;
    v12.uniformAndStorageBuffer8BitAccess = VK_TRUE; v12.shaderFloat16 = VK_TRUE;
    v12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceVulkanMemoryModelFeatures vmm = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES };
    vmm.pNext = &v12; vmm.vulkanMemoryModel = VK_TRUE; vmm.vulkanMemoryModelDeviceScope = VK_TRUE;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
    cm.pNext = &vmm; cm.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 }; f2.pNext = &cm;

    const char* devExt[] = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = devExt;
    VC(vkCreateDevice(phys, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);
    pGetBufAddr = (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(dev, "vkGetBufferDeviceAddress");
    if (!pGetBufAddr) { fprintf(stderr,"no vkGetBufferDeviceAddress\n"); return 1; }

    // buffers
    VkBufferUsageFlags U = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkDeviceSize asz=(VkDeviceSize)M*K, bsz=(VkDeviceSize)K*N, btsz=(VkDeviceSize)N*K, csz=(VkDeviceSize)M*N*4;
    GpuBuf A  = make_buf(asz,  U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    GpuBuf B  = make_buf(bsz,  U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    GpuBuf Bt = make_buf(btsz, U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    GpuBuf C  = make_buf(csz,  U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    GpuBuf Af = make_buf(asz*2, U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    GpuBuf Bf = make_buf(bsz*2, U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    GpuBuf Cf = make_buf(csz,   U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    int8_t* ha = malloc(asz); int8_t* hb = malloc(bsz); int8_t* hbt = malloc(btsz);
    uint16_t* hfa = malloc(asz*2); uint16_t* hfb = malloc(bsz*2);
    for (VkDeviceSize i=0;i<asz;i++) ha[i]=(int8_t)(((i*7+3)%251)-125);
    for (VkDeviceSize k=0;k<(VkDeviceSize)K;k++) for (VkDeviceSize j=0;j<(VkDeviceSize)N;j++){
        int8_t v=(int8_t)(((k*131+j*17+5)%251)-125); hb[k*N+j]=v; hbt[(VkDeviceSize)j*K+k]=v; }
    for (VkDeviceSize i=0;i<asz;i++) hfa[i]=f2h((float)(((int)(i*3+1)%9)-4));
    for (VkDeviceSize i=0;i<bsz;i++) hfb[i]=f2h((float)(((int)(i*5+2)%9)-4));
    upload(&A, ha, asz); upload(&B, hb, bsz); upload(&Bt, hbt, btsz);
    upload(&Af, hfa, asz*2); upload(&Bf, hfb, bsz*2);

    // kernels
    Kern kernels[] = {
        { "XMX-i8",  "bin/gemm_xmx.spv",     0, 0, 0,0, (M+15)/16, (N+31)/32 },
        { "SIMT-i8", "bin/gemm_simt.spv",    0, 1, 0,0, (N+15)/16, (M+15)/16 },
        { "DP4A-i8", "bin/gemm_dp4a.spv",    0, 1, 0,0, (N+15)/16, (M+15)/16 },
        { "XMX-f16", "bin/gemm_f16_xmx.spv", 1, 0, 0,0, (M+15)/16, (N+31)/32 },
        { "SIMT-f16","bin/gemm_f16_simt.spv",1, 1, 0,0, (N+15)/16, (M+15)/16 },
    };
    int nk = sizeof(kernels)/sizeof(kernels[0]);
    for (int i=0;i<nk;i++) build_kernel(&kernels[i]);

    VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2;
    VkQueryPool tqp; VC(vkCreateQueryPool(dev, &qpi, NULL, &tqp));
    VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence fence; VC(vkCreateFence(dev, &fi, NULL, &fence));

    double ops = 2.0*(double)M*(double)N*(double)K;
    printf("M=N=K=%d   ops/dispatch = %.3g\n\n", M, ops);
    printf("%-9s %12s  %10s   %10s\n", "kernel", "ms/disp", "GOPS", "vs XMX-i8");

    double xmx_i8 = 0.0;
    for (int i=0;i<nk;i++) {
        Kern* k = &kernels[i];
        GpuBuf* AA = k->f16 ? &Af : &A;
        GpuBuf* BB = k->f16 ? &Bf : (strcmp(k->name,"DP4A-i8")==0 ? &Bt : &B);
        GpuBuf* CC = k->f16 ? &Cf : &C;

        // correctness
        DispArgs d = { k, AA, BB, CC, M,N,K, 1, tqp };
        run_cmd(rec_dispatch, &d, fence);
        int ok=0, n=200, bad=0;
        if (k->f16) {
            float* hc = malloc(csz); readback(CC, hc, csz);
            for (int t=0;t<n;t++){ int r=(t*97)%M, c=(t*131)%N; double s=0;
                for (int kk=0;kk<K;kk++) s += (double)h2f(hfa[(VkDeviceSize)r*K+kk]) * h2f(hfb[(VkDeviceSize)kk*N+c]);
                double got = hc[(VkDeviceSize)r*N+c]; double tol = 1e-2*((s<0)?-s:s)+1e-3;
                if ((got-s<tol)&&(s-got<tol)) ok++; else if(bad++<2) fprintf(stderr,"  [%s] @(%d,%d) got %.3f want %.3f\n",k->name,r,c,got,s); }
            free(hc);
        } else {
            int32_t* hc = malloc(csz); readback(CC, hc, csz);
            for (int t=0;t<n;t++){ int r=(t*97)%M, c=(t*131)%N; long s=0;
                for (int kk=0;kk<K;kk++) s += (long)ha[(VkDeviceSize)r*K+kk]*(long)hb[(VkDeviceSize)kk*N+c];
                if (hc[(VkDeviceSize)r*N+c]==(int32_t)s) ok++; else if(bad++<2) fprintf(stderr,"  [%s] @(%d,%d) got %d want %ld\n",k->name,r,c,hc[(VkDeviceSize)r*N+c],s); }
            free(hc);
        }
        if (ok!=n) { printf("%-9s correctness FAILED (%d/%d)\n", k->name, ok, n); continue; }

        // perf (GPU timestamps)
        int reps = (int)(4.0e10/ops); if(reps<1)reps=1; if(reps>50)reps=50;
        d.reps = reps;
        run_cmd(rec_dispatch, &d, fence); // warmup
        double best = 1e30;
        for (int s=0;s<5;s++){
            run_cmd(rec_dispatch, &d, fence);
            uint64_t ts[2]={0,0}; VC(vkGetQueryPoolResults(dev, tqp, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
            double ms = (double)(ts[1]-ts[0]) * timestampPeriod / 1e6 / reps;
            if (ms < best) best = ms;
        }
        double gops = ops/best/1e6;
        if (i==0) xmx_i8 = gops;
        printf("%-9s %12.3f  %10.1f   %9.2fx\n", k->name, best, gops, xmx_i8>0? gops/xmx_i8 : 0.0);
    }
    printf("\n(correctness: 200/%d sampled outputs verified per kernel)\n", 200);
    return 0;
}
