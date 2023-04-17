<script lang="ts">
	import Chatbot from '$lib/components/chatbot.svelte';
	export let data;
</script>

<Chatbot collection={data?.collection || ''} index="documents" collections={data.collections} />
